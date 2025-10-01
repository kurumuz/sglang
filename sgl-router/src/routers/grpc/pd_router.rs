// PD (Prefill-Decode) gRPC Router Implementation

use crate::config::types::RetryConfig;
use crate::core::{ConnectionMode, Worker, WorkerRegistry, WorkerType};
use crate::grpc_client::proto;
use crate::policies::PolicyRegistry;
use crate::protocols::spec::{
    ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, EmbeddingRequest, FunctionCallResponse, GenerateRequest, RerankRequest,
    ResponsesGetParams, ResponsesRequest, ToolCall, ToolChoice, ToolChoiceValue, Usage,
};
use crate::reasoning_parser::ParserFactory;
use crate::routers::http::pd_types::generate_room_id;
use crate::routers::{grpc, RouterTrait};
use crate::server::AppContext;
use crate::tokenizer::traits::Tokenizer;
use crate::tokenizer::{SequenceDecoderOutput, StopSequenceDecoder};
use crate::tool_parser::ToolParserFactory;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use grpc::utils;
use serde_json::Value;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, warn};
use uuid::Uuid;

/// gRPC PD (Prefill-Decode) router implementation for SGLang
#[allow(dead_code)] // Fields will be used once implementation is complete
pub struct GrpcPDRouter {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    tokenizer: Arc<dyn Tokenizer>,
    reasoning_parser_factory: ParserFactory,
    tool_parser_factory: ToolParserFactory,
    dp_aware: bool,
    api_key: Option<String>,
    retry_config: RetryConfig,
}

impl GrpcPDRouter {
    /// Create a new gRPC PD router
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        // Get registries from context
        let worker_registry = ctx.worker_registry.clone();
        let policy_registry = ctx.policy_registry.clone();

        // Extract necessary components from context
        let tokenizer = ctx
            .tokenizer
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires tokenizer".to_string())?
            .clone();
        let reasoning_parser_factory = ctx
            .reasoning_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires reasoning parser factory".to_string())?
            .clone();
        let tool_parser_factory = ctx
            .tool_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires tool parser factory".to_string())?
            .clone();

        Ok(GrpcPDRouter {
            worker_registry,
            policy_registry,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_factory,
            dp_aware: ctx.router_config.dp_aware,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    /// Select a prefill-decode worker pair using load balancing policies
    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let effective_model_id = if !self.dp_aware { None } else { model_id };

        debug!(
            "Selecting PD pair: dp_aware={}, model_id={:?}, effective_model_id={:?}",
            self.dp_aware, model_id, effective_model_id
        );

        // Get prefill workers
        let prefill_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model_fast(model)
                .into_iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .collect()
        } else {
            self.worker_registry.get_workers_filtered(
                None,
                Some(WorkerType::Prefill {
                    bootstrap_port: None,
                }),
                Some(ConnectionMode::Grpc { port: None }),
                true, // only healthy workers
            )
        };

        // Get decode workers
        let decode_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model_fast(model)
                .into_iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .collect()
        } else {
            self.worker_registry.get_workers_filtered(
                None,
                Some(WorkerType::Decode),
                Some(ConnectionMode::Grpc { port: None }),
                true, // only healthy workers
            )
        };

        if prefill_workers.is_empty() {
            return Err("No healthy prefill workers available".to_string());
        }
        if decode_workers.is_empty() {
            return Err("No healthy decode workers available".to_string());
        }

        debug!(
            "Found {} prefill workers and {} decode workers",
            prefill_workers.len(),
            decode_workers.len()
        );

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        let prefill_idx = prefill_policy
            .select_worker(&prefill_workers, request_text)
            .ok_or_else(|| "Failed to select prefill worker".to_string())?;

        let decode_idx = decode_policy
            .select_worker(&decode_workers, request_text)
            .ok_or_else(|| "Failed to select decode worker".to_string())?;

        let prefill = prefill_workers[prefill_idx].clone();
        let decode = decode_workers[decode_idx].clone();

        debug!(
            "Selected PD pair: prefill={}, decode={}",
            prefill.url(),
            decode.url()
        );

        Ok((prefill, decode))
    }

    /// Inject bootstrap metadata into a protobuf GenerateRequest
    fn inject_bootstrap_metadata(
        request: &mut proto::GenerateRequest,
        prefill_worker: &dyn Worker,
    ) -> Result<(), String> {
        let hostname = prefill_worker.bootstrap_host();
        let bootstrap_port = prefill_worker.bootstrap_port().unwrap_or(8998);

        let room_id = generate_room_id();

        // Create DisaggregatedParams
        let disagg_params = proto::DisaggregatedParams {
            bootstrap_host: hostname.to_string(),
            bootstrap_port: bootstrap_port as i32,
            bootstrap_room: room_id as i32,
        };

        // Inject metadata
        request.disaggregated_params = Some(disagg_params);

        debug!(
            "Injected bootstrap metadata: host={}, port={}, room={}",
            hostname, bootstrap_port, room_id
        );

        Ok(())
    }

    /// Main route_chat implementation with PD dual dispatch
    async fn route_chat_impl(
        &self,
        _headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing chat completion request for model: {:?} (PD mode)",
            model_id
        );

        // Step 1: Filter tools if needed for allowed_tools or specific function
        let body_ref = utils::filter_tools_for_request(body);

        // Step 2: Process messages and apply chat template
        let processed_messages = match utils::process_chat_messages(&body_ref, &*self.tokenizer) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!("Failed to process chat messages: {}", e);
                return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
            }
        };

        // Step 3: Tokenize the processed text
        let encoding = match self.tokenizer.encode(&processed_messages.text) {
            Ok(encoding) => encoding,
            Err(e) => {
                error!("Tokenization failed: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Tokenization failed: {}", e),
                )
                    .into_response();
            }
        };

        // Step 4: Build tool constraints if needed
        // body_ref already has filtered tools if needed
        let tool_call_constraint = body_ref.tools.as_ref().and_then(|tools| {
            utils::generate_tool_constraints(tools, &body.tool_choice, &body.model)
        });

        let token_ids = encoding.token_ids().to_vec();
        debug!("Tokenized {} tokens from input", token_ids.len());

        // Step 5: Select prefill-decode worker pair
        let (prefill_worker, decode_worker) = match self
            .select_pd_pair(Some(&processed_messages.text), model_id)
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                warn!("Failed to select PD worker pair: {}", e);
                return (StatusCode::SERVICE_UNAVAILABLE, e).into_response();
            }
        };

        debug!(
            "Selected PD pair: prefill={}, decode={}",
            prefill_worker.url(),
            decode_worker.url()
        );

        // Step 6: Get gRPC clients for both workers
        let prefill_client = match utils::get_grpc_client_from_worker(&prefill_worker).await {
            Ok(client) => client,
            Err(response) => return response,
        };

        let decode_client = match utils::get_grpc_client_from_worker(&decode_worker).await {
            Ok(client) => client,
            Err(response) => return response,
        };

        // Step 7: Build the base gRPC request
        let request_id = format!("chatcmpl-{}", Uuid::new_v4());
        let mut request = match prefill_client.build_generate_request(
            request_id.clone(),
            &body_ref,
            processed_messages.text.clone(),
            token_ids,
            processed_messages.multimodal_inputs,
            tool_call_constraint,
        ) {
            Ok(request) => request,
            Err(e) => {
                error!("Failed to build gRPC request: {}", e);
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid request parameters: {}", e),
                )
                    .into_response();
            }
        };

        // Step 8: Inject bootstrap metadata into the request
        if let Err(e) = Self::inject_bootstrap_metadata(&mut request, &*prefill_worker) {
            error!("Failed to inject bootstrap metadata: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
        }

        // Step 9: Handle streaming vs non-streaming
        if body.stream {
            self.handle_streaming_chat(prefill_client, decode_client, request, body)
                .await
        } else {
            self.handle_non_streaming_chat(prefill_client, decode_client, request, body)
                .await
        }
    }

    /// Submit request and handle streaming response for chat completions (PD mode)
    async fn handle_streaming_chat(
        &self,
        mut _prefill_client: crate::grpc_client::SglangSchedulerClient,
        mut _decode_client: crate::grpc_client::SglangSchedulerClient,
        _request: proto::GenerateRequest,
        _original_request: &ChatCompletionRequest,
    ) -> Response {
        // TODO: Implement streaming response for PD mode
        (
            StatusCode::NOT_IMPLEMENTED,
            "Streaming not yet implemented for PD mode",
        )
            .into_response()
    }

    /// Submit request and handle non-streaming response for chat completions (PD mode)
    async fn handle_non_streaming_chat(
        &self,
        mut prefill_client: crate::grpc_client::SglangSchedulerClient,
        mut decode_client: crate::grpc_client::SglangSchedulerClient,
        request: proto::GenerateRequest,
        original_request: &ChatCompletionRequest,
    ) -> Response {
        // Step 1: Create stop decoder
        let mut stop_decoder = utils::create_stop_decoder(
            &self.tokenizer,
            original_request.stop.as_ref(),
            original_request.stop_token_ids.as_ref(),
            original_request.skip_special_tokens,
            original_request.no_stop_trim,
        );

        // Step 2: Send requests in parallel
        debug!("Sending concurrent requests to prefill and decode workers");
        let prefill_request = request.clone();
        let decode_request = request;

        let (prefill_result, decode_result) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Step 3: Process prefill stream in parallel - if it fails, assume decode fails
        let prefill_stream = match prefill_result {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to start prefill generation: {}", e);
                return utils::internal_error_message(format!(
                    "Prefill worker failed to start: {}",
                    e
                ));
            }
        };

        let decode_stream = match decode_result {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to start decode generation: {}", e);
                return utils::internal_error_message(format!(
                    "Decode worker failed to start: {}",
                    e
                ));
            }
        };

        // Collect prefill response (for input_logprobs if requested)
        let prefill_responses =
            match utils::collect_stream_responses(prefill_stream, "Prefill").await {
                Ok(responses) => responses,
                Err(error_response) => return error_response,
            };

        // Extract input_logprobs from prefill response if available
        let prefill_input_logprobs = prefill_responses
            .first()
            .and_then(|r| r.input_logprobs.clone());

        // Step 4: Process decode stream (collect all responses for n>1 support)
        let all_responses = match utils::collect_stream_responses(decode_stream, "Decode").await {
            Ok(responses) => responses,
            Err(error_response) => return error_response,
        };

        if all_responses.is_empty() {
            return utils::internal_error_static("No responses from decode worker");
        }

        // Process each response into a ChatChoice
        let mut choices = Vec::new();
        for (index, complete) in all_responses.iter().enumerate() {
            // Merge prefill input_logprobs if available and requested
            let mut complete_with_logprobs = complete.clone();
            if prefill_input_logprobs.is_some() && original_request.logprobs {
                complete_with_logprobs.input_logprobs = prefill_input_logprobs.clone();
            }

            match self
                .process_single_choice(
                    &complete_with_logprobs,
                    index,
                    original_request,
                    &mut stop_decoder,
                )
                .await
            {
                Ok(choice) => choices.push(choice),
                Err(e) => {
                    return utils::internal_error_message(format!(
                        "Failed to process choice {}: {}",
                        index, e
                    ));
                }
            }
        }

        // Aggregate usage information from all responses
        let total_prompt_tokens: u32 = all_responses.iter().map(|r| r.prompt_tokens as u32).sum();
        let total_completion_tokens: u32 = all_responses
            .iter()
            .map(|r| r.completion_tokens as u32)
            .sum();

        let usage = Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
            completion_tokens_details: None,
        };

        // Build final ChatCompletionResponse
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: original_request.model.clone(),
            choices,
            usage: Some(usage),
            system_fingerprint: None,
        };

        // Serialize and return JSON response
        Json(response).into_response()
    }

    /// Process a single GenerateComplete response into a ChatChoice
    async fn process_single_choice(
        &self,
        complete: &proto::GenerateComplete,
        index: usize,
        original_request: &ChatCompletionRequest,
        stop_decoder: &mut StopSequenceDecoder,
    ) -> Result<ChatChoice, String> {
        stop_decoder.reset();
        // Decode tokens
        let outputs = stop_decoder
            .process_tokens(&complete.output_ids)
            .map_err(|e| format!("Failed to process tokens: {}", e))?;

        // Accumulate text with early breaks
        let mut final_text = String::new();
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => final_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    final_text.push_str(&t);
                    break;
                }
                SequenceDecoderOutput::Stopped => break,
                SequenceDecoderOutput::Held => {}
            }
        }

        // Flush remaining text
        if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
            final_text.push_str(&t);
        }

        // Step 1: Handle reasoning content parsing
        let mut reasoning_text: Option<String> = None;
        let mut processed_text = final_text;

        // Check if reasoning parsing is enabled and separate_reasoning is requested
        if original_request.separate_reasoning {
            let pooled_parser = self
                .reasoning_parser_factory
                .get_pooled(&original_request.model);

            let mut parser = pooled_parser
                .lock()
                .map_err(|e| format!("Failed to acquire reasoning parser lock: {}", e))?;
            match parser.detect_and_parse_reasoning(&processed_text) {
                Ok(result) => {
                    if !result.reasoning_text.is_empty() {
                        reasoning_text = Some(result.reasoning_text);
                    }
                    processed_text = result.normal_text;
                }
                Err(e) => {
                    return Err(format!("Reasoning parsing error: {}", e));
                }
            }
        }

        // Step 2: Handle tool call parsing
        let mut tool_calls: Option<Vec<ToolCall>> = None;

        // Check if tool calls should be processed
        let tool_choice_enabled = !matches!(
            &original_request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        if tool_choice_enabled && original_request.tools.is_some() {
            // Check if JSON schema constraint was used (specific function or required mode)
            let used_json_schema = match &original_request.tool_choice {
                Some(ToolChoice::Function { .. }) => true,
                Some(ToolChoice::Value(ToolChoiceValue::Required)) => true,
                Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
                _ => false,
            };

            if used_json_schema {
                (tool_calls, processed_text) = utils::parse_json_schema_response(
                    &processed_text,
                    &original_request.tool_choice,
                );
            } else {
                (tool_calls, processed_text) = self
                    .parse_with_model_parser(&processed_text, &original_request.model)
                    .await;
            }
        }

        // Step 3: Use finish reason directly from proto (already OpenAI-compatible string)
        let finish_reason_str = &complete.finish_reason;

        // Override finish reason if we have tool calls
        let final_finish_reason_str = if tool_calls.is_some() {
            "tool_calls"
        } else {
            finish_reason_str
        };

        // Extract matched_stop information from proto
        let matched_stop = match &complete.matched_stop {
            Some(proto::generate_complete::MatchedStop::MatchedTokenId(token_id)) => {
                Some(Value::Number(serde_json::Number::from(*token_id)))
            }
            Some(proto::generate_complete::MatchedStop::MatchedStopStr(stop_str)) => {
                Some(Value::String(stop_str.clone()))
            }
            None => None,
        };

        // Step 4: Convert output logprobs if present
        // Note: complete.input_logprobs exists in proto but is not used for chat completions
        //       (input logprobs are only used in /v1/completions endpoint with echo=true)
        let logprobs = if let Some(proto_logprobs) = &complete.output_logprobs {
            match self.convert_proto_to_openai_logprobs(proto_logprobs) {
                Ok(logprobs) => Some(logprobs),
                Err(e) => {
                    error!("Failed to convert logprobs: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 5: Build ChatCompletionMessage (proper response message type)
        let chat_message = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: if processed_text.is_empty() {
                None
            } else {
                Some(processed_text)
            },
            tool_calls,
            reasoning_content: reasoning_text,
        };

        // Step 6: Build ChatChoice
        let choice = ChatChoice {
            index: index as u32,
            message: chat_message,
            logprobs,
            finish_reason: Some(final_finish_reason_str.to_string()),
            matched_stop,
            hidden_states: None,
        };

        Ok(choice)
    }

    /// Convert proto LogProbs to OpenAI ChatLogProbs format
    /// Note: Always decodes with skip_special_tokens=false to show actual tokens generated
    fn convert_proto_to_openai_logprobs(
        &self,
        proto_logprobs: &proto::OutputLogProbs,
    ) -> Result<crate::protocols::spec::ChatLogProbs, String> {
        let mut content_items = Vec::new();

        // Decode token IDs to text (always with skip_special_tokens=false for logprobs)
        let token_texts: Vec<String> = proto_logprobs
            .token_ids
            .iter()
            .map(|&token_id| {
                self.tokenizer
                    .decode(&[token_id as u32], false)
                    .unwrap_or_else(|_| format!("<token_{}>", token_id))
            })
            .collect();

        // Build ChatLogProbsContent for each token
        for (i, &logprob) in proto_logprobs.token_logprobs.iter().enumerate() {
            let token_text = token_texts.get(i).cloned().unwrap_or_default();
            let bytes = Some(token_text.as_bytes().to_vec());

            // Build top_logprobs for this position
            let mut top_logprobs = Vec::new();
            if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
                // Decode top token IDs (always with skip_special_tokens=false)
                let top_token_texts: Vec<String> = top_logprobs_entry
                    .token_ids
                    .iter()
                    .map(|&tid| {
                        self.tokenizer
                            .decode(&[tid as u32], false)
                            .unwrap_or_else(|_| format!("<token_{}>", tid))
                    })
                    .collect();

                for (j, (&top_logprob, &_top_token_id)) in top_logprobs_entry
                    .values
                    .iter()
                    .zip(top_logprobs_entry.token_ids.iter())
                    .enumerate()
                {
                    if let Some(top_token_text) = top_token_texts.get(j) {
                        top_logprobs.push(crate::protocols::spec::TopLogProb {
                            token: top_token_text.clone(),
                            logprob: top_logprob,
                            bytes: Some(top_token_text.as_bytes().to_vec()),
                        });
                    }
                }
            }

            content_items.push(crate::protocols::spec::ChatLogProbsContent {
                token: token_text,
                logprob,
                bytes,
                top_logprobs,
            });
        }

        Ok(crate::protocols::spec::ChatLogProbs::Detailed {
            content: (!content_items.is_empty()).then_some(content_items),
        })
    }

    /// Parse tool calls using model-specific parser
    async fn parse_with_model_parser(
        &self,
        processed_text: &str,
        model: &str,
    ) -> (Option<Vec<ToolCall>>, String) {
        // Get pooled parser for this model
        let pooled_parser = self.tool_parser_factory.get_pooled(model);

        // Check format detection first
        let can_parse = {
            let parser = pooled_parser.lock().await;
            parser.detect_format(processed_text)
            // Lock is dropped here
        };

        if !can_parse {
            return (None, processed_text.to_string());
        }

        // Lock again for async parsing
        let result = {
            let parser = pooled_parser.lock().await;
            parser.parse_complete(processed_text).await
            // Lock is dropped here
        };

        match result {
            Ok((normal_text, parsed_tool_calls)) => {
                if parsed_tool_calls.is_empty() {
                    return (None, normal_text);
                }

                let spec_tool_calls = parsed_tool_calls
                    .into_iter()
                    .map(|tc| ToolCall {
                        id: tc.id,
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: tc.function.name,
                            arguments: Some(
                                serde_json::to_string(&tc.function.arguments)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            ),
                        },
                    })
                    .collect();
                (Some(spec_tool_calls), normal_text)
            }
            Err(e) => {
                error!("Tool call parsing error: {}", e);
                (None, processed_text.to_string())
            }
        }
    }
}

impl std::fmt::Debug for GrpcPDRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefill_workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Prefill {
                bootstrap_port: None,
            }),
            Some(crate::core::ConnectionMode::Grpc { port: None }),
            false,
        );
        let decode_workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Decode),
            Some(crate::core::ConnectionMode::Grpc { port: None }),
            false,
        );
        f.debug_struct("GrpcPDRouter")
            .field("prefill_workers_count", &prefill_workers.len())
            .field("decode_workers_count", &decode_workers.len())
            .field("dp_aware", &self.dp_aware)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcPDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // TODO: Implement actual generation test for gRPC PD mode
        (
            StatusCode::NOT_IMPLEMENTED,
            "Health generate not yet implemented for gRPC PD",
        )
            .into_response()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_chat_impl(headers, body, model_id).await
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    fn router_type(&self) -> &'static str {
        "grpc_pd"
    }
}
