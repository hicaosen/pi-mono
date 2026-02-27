/**
 * iFlow provider - OpenAI-compatible API with iFlow-specific headers and extensions.
 * Reuses openai-completions.ts logic with iFlow-specific signature calculation and headers.
 */

import * as crypto from "crypto";
import OpenAI from "openai";
import { getEnvApiKey } from "../env-api-keys.js";
import { calculateCost } from "../models.js";
import type {
	AssistantMessage,
	Context,
	Model,
	OpenAICompletionsCompat,
	SimpleStreamOptions,
	StopReason,
	StreamFunction,
	StreamOptions,
	TextContent,
	ThinkingContent,
	ToolCall,
} from "../types.js";
import { AssistantMessageEventStream } from "../utils/event-stream.js";
import { parseStreamingJson } from "../utils/json-parse.js";
import { convertMessages } from "./openai-completions.js";
import { buildBaseOptions, clampReasoning } from "./simple-options.js";

const IFLOW_USER_AGENT = "iFlow-Cli";

export interface IflowOptions extends StreamOptions {
	toolChoice?: "auto" | "none" | "required" | { type: "function"; function: { name: string } };
	reasoningEffort?: "minimal" | "low" | "medium" | "high" | "xhigh";
	/** iFlow extension field. If omitted, maxTokens is mirrored into max_new_tokens. */
	maxNewTokens?: number;
	/** iFlow extension fields for thinking/reasoning compatibility. */
	thinking?: { type: "enabled" | "disabled" } | boolean;
	enableThinking?: boolean;
	thinkingMode?: boolean;
	/** Enable thinking/reasoning. Can be boolean or thinking level. */
	reasoning?: boolean | "minimal" | "low" | "medium" | "high" | "xhigh";
	chatTemplateKwargs?: Record<string, unknown>;
	extendFields?: Record<string, unknown>;
}

/** Generate iFlow signature: HMAC_SHA256_HEX(apiKey, `${userAgent}:${sessionId}:${timestampMs}`) */
function generateSignature(userAgent: string, sessionId: string, timestampMs: number, apiKey: string): string {
	const stringToSign = `${userAgent}:${sessionId}:${timestampMs}`;
	return crypto.createHmac("sha256", apiKey).update(stringToSign, "utf8").digest("hex");
}

/** Generate a session ID in iFlow format: session-{uuid} */
function generateSessionId(): string {
	return `session-${crypto.randomUUID()}`;
}

/** Normalize model IDs for iFlow-specific matching logic. */
function normalizeIflowModelId(modelId: string): string {
	return modelId.trim().toLowerCase();
}

function isDeepSeekV32Model(modelId: string): boolean {
	const normalized = normalizeIflowModelId(modelId);
	return normalized === "deepseek-v3.2" || normalized === "deepseek-v3.2-chat";
}

function isKimiK25Model(modelId: string): boolean {
	const normalized = normalizeIflowModelId(modelId);
	return normalized === "kimi-k2.5" || normalized.startsWith("kimi-k2.5-");
}

/** Models that support setThink behavior in iflow.js. */
function supportsSetThink(modelId: string): boolean {
	const normalized = normalizeIflowModelId(modelId);
	return (
		normalized === "glm-4.7" ||
		normalized === "glm-5" ||
		normalized === "kimi-k2-thinking" ||
		isDeepSeekV32Model(normalized) ||
		isKimiK25Model(normalized)
	);
}

/** Determine if thinking/reasoning is enabled based on options. */
function isThinkingEnabled(options: IflowOptions | undefined): boolean {
	if (options?.thinking !== undefined) {
		if (typeof options.thinking === "object" && "type" in options.thinking) {
			return options.thinking.type === "enabled";
		}
		return Boolean(options.thinking);
	}
	if (options?.enableThinking !== undefined) return options.enableThinking;
	if (options?.thinkingMode !== undefined) return options.thinkingMode;
	if (options?.reasoning !== undefined) return Boolean(options.reasoning);
	return false;
}

function getReasoningLevel(options: IflowOptions | undefined): IflowOptions["reasoningEffort"] | undefined {
	if (typeof options?.reasoning === "string") {
		return options.reasoning;
	}
	return options?.reasoningEffort;
}

/** iFlow compatibility settings - similar to z.ai with thinking: { type: "enabled" | "disabled" } */
const IFLOW_COMPAT: Required<OpenAICompletionsCompat> = {
	supportsStore: false,
	supportsDeveloperRole: false,
	supportsReasoningEffort: true,
	supportsUsageInStreaming: true,
	maxTokensField: "max_tokens",
	requiresToolResultName: false,
	requiresAssistantAfterToolResult: false,
	requiresThinkingAsText: false,
	requiresMistralToolIds: false,
	thinkingFormat: "zai", // Use thinking: { type: "enabled" | "disabled" } like z.ai
	openRouterRouting: {},
	vercelGatewayRouting: {},
	supportsStrictMode: true,
};

/**
 * Apply model-specific parameters based on iFlow documentation.
 * @internal Exported for testing purposes
 */
export function applyModelSpecificParams(
	model: Model<"iflow-completions">,
	options: IflowOptions | undefined,
	context: Context,
): Record<string, unknown> {
	const thinkingEnabled = isThinkingEnabled(options);
	const reasoningLevel = getReasoningLevel(options);
	const params: Record<string, unknown> = {};
	const hasTools = Boolean(context.tools && context.tools.length > 0);
	const normalizedModelId = normalizeIflowModelId(model.id);

	if (normalizedModelId === "iflow-rome-30ba3b") {
		params.temperature = 0.7;
		params.top_p = 0.8;
		params.top_k = 20;
		return params;
	}

	if (normalizedModelId === "glm-4.7") {
		params.temperature = 1;
		params.top_p = 0.95;
		params.chat_template_kwargs = { enable_thinking: thinkingEnabled };
		return params;
	}

	if (isDeepSeekV32Model(normalizedModelId)) {
		if (thinkingEnabled) {
			params.model = "deepseek-v3.2-reasoner";
			params.thinking_mode = true;
			if (reasoningLevel !== "low") {
				params.reasoning = true;
			}
		}
		return params;
	}

	if (normalizedModelId === "glm-5") {
		params.temperature = 1;
		params.top_p = 0.95;
		params.chat_template_kwargs = { enable_thinking: thinkingEnabled };
		params.enable_thinking = thinkingEnabled;
		params.thinking = { type: thinkingEnabled ? "enabled" : "disabled" };
		return params;
	}

	if (normalizedModelId === "kimi-k2-thinking") {
		if (thinkingEnabled) {
			params.thinking_mode = true;
		}
		return params;
	}

	if (isKimiK25Model(normalizedModelId)) {
		params.temperature = undefined;
		params.top_p = 0.95;
		params.n = 1;
		params.presence_penalty = 0;
		params.frequency_penalty = 0;
		params.max_tokens = options?.maxTokens ?? model.maxTokens;
		params.thinking = thinkingEnabled ? { type: "enabled" } : { type: "disabled" };
		if (thinkingEnabled && hasTools) {
			const currentChoice = options?.toolChoice;
			if (!currentChoice || (currentChoice !== "auto" && currentChoice !== "none")) {
				params.tool_choice = "auto";
			}
		}
		return params;
	}

	return params;
}

/**
 * Convert messages for iFlow API using openai-completions convertMessages.
 * @internal Exported for testing purposes
 */
export function convertIflowMessages(
	model: Model<"iflow-completions">,
	context: Context,
): Array<Record<string, unknown>> {
	const openaiModel = model as unknown as Model<"openai-completions">;
	return convertMessages(openaiModel, context, IFLOW_COMPAT) as unknown as Array<Record<string, unknown>>;
}

function mapStopReason(reason: string | null): StopReason {
	if (reason === null) return "stop";
	switch (reason) {
		case "stop":
			return "stop";
		case "length":
			return "length";
		case "function_call":
		case "tool_calls":
			return "toolUse";
		case "content_filter":
			return "error";
		default:
			return "stop";
	}
}

function createClient(
	model: Model<"iflow-completions">,
	apiKey: string,
	sessionId: string,
	timestampMs: number,
	signature: string,
	optionsHeaders?: Record<string, string>,
) {
	const headers: Record<string, string> = {
		...model.headers,
		"user-agent": IFLOW_USER_AGENT,
		"session-id": sessionId,
		"x-iflow-signature": signature,
		"x-iflow-timestamp": String(timestampMs),
		...(optionsHeaders ?? {}),
	};

	return new OpenAI({
		apiKey,
		baseURL: model.baseUrl,
		dangerouslyAllowBrowser: true,
		defaultHeaders: headers,
	});
}

export const streamIflow: StreamFunction<"iflow-completions", IflowOptions> = (
	model: Model<"iflow-completions">,
	context: Context,
	options?: IflowOptions,
): AssistantMessageEventStream => {
	const stream = new AssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const apiKey = options?.apiKey || getEnvApiKey(model.provider);
			if (!apiKey) {
				throw new Error("iFlow API key is required. Set IFLOW_API_KEY environment variable or pass apiKey.");
			}

			// Map pi sessionId to iflow session-id
			const sessionId = options?.sessionId || generateSessionId();
			const timestampMs = Date.now();
			const signature = generateSignature(IFLOW_USER_AGENT, sessionId, timestampMs, apiKey);
			const client = createClient(model, apiKey, sessionId, timestampMs, signature, options?.headers);

			// Build params using openai-completions convertMessages with iFlow compat
			const openaiModel = model as unknown as Model<"openai-completions">;
			const messages = convertMessages(openaiModel, context, IFLOW_COMPAT);

			const params: Record<string, unknown> = {
				model: model.id,
				messages,
				stream: true,
				stream_options: { include_usage: true },
			};

			if (options?.maxTokens) {
				params.max_tokens = options.maxTokens;
			}

			if (options?.temperature !== undefined) {
				params.temperature = options.temperature;
			}

			if (context.tools && context.tools.length > 0) {
				params.tools = context.tools.map((tool) => ({
					type: "function",
					function: {
						name: tool.name,
						description: tool.description,
						parameters: tool.parameters,
						strict: false,
					},
				}));
			}

			if (options?.toolChoice) {
				params.tool_choice = options.toolChoice;
			}

			// Handle extend_fields - include sessionId for tracking
			const extendFields: Record<string, unknown> = {
				...(options?.extendFields ?? {}),
			};
			if (!("sessionId" in extendFields)) {
				extendFields.sessionId = sessionId;
			}
			if (Object.keys(extendFields).length > 0) {
				params.extend_fields = extendFields;
			}

			// Apply model-specific parameters
			const modelSpecificParams = applyModelSpecificParams(model, options, context);
			Object.assign(params, modelSpecificParams);

			options?.onPayload?.(params);

			const openaiStream = await client.chat.completions.create(
				params as unknown as OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming,
				{ signal: options?.signal },
			);

			stream.push({ type: "start", partial: output });

			let currentBlock: TextContent | ThinkingContent | (ToolCall & { partialArgs?: string }) | null = null;
			const blocks = output.content;
			const blockIndex = () => blocks.length - 1;

			const finishCurrentBlock = (block?: typeof currentBlock) => {
				if (!block) return;

				if (block.type === "text") {
					stream.push({
						type: "text_end",
						contentIndex: blockIndex(),
						content: block.text,
						partial: output,
					});
				} else if (block.type === "thinking") {
					stream.push({
						type: "thinking_end",
						contentIndex: blockIndex(),
						content: block.thinking,
						partial: output,
					});
				} else if (block.type === "toolCall") {
					block.arguments = parseStreamingJson(block.partialArgs);
					delete block.partialArgs;
					stream.push({
						type: "toolcall_end",
						contentIndex: blockIndex(),
						toolCall: block,
						partial: output,
					});
				}
			};

			for await (const chunk of openaiStream) {
				// Handle usage (iFlow format)
				if (chunk.usage) {
					const usage = chunk.usage as any;
					const promptTokens = usage.prompt_tokens || 0;
					const completionTokens = usage.completion_tokens || 0;
					const totalTokens = usage.total_tokens || promptTokens + completionTokens;
					const cacheRead = usage.cache_read_input_tokens || 0;
					const cacheWrite = usage.cache_creation_input_tokens || 0;

					output.usage = {
						input: promptTokens,
						output: completionTokens,
						cacheRead,
						cacheWrite,
						totalTokens,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					};
					calculateCost(model, output.usage);
				}

				const choice = chunk.choices[0];
				if (!choice) continue;

				if (choice.finish_reason) {
					output.stopReason = mapStopReason(choice.finish_reason);
				}

				if (!choice.delta) continue;

				// Handle text content
				if (
					choice.delta.content !== null &&
					choice.delta.content !== undefined &&
					choice.delta.content.length > 0
				) {
					if (!currentBlock || currentBlock.type !== "text") {
						finishCurrentBlock(currentBlock);
						currentBlock = { type: "text", text: "" };
						output.content.push(currentBlock);
						stream.push({ type: "text_start", contentIndex: blockIndex(), partial: output });
					}

					if (currentBlock.type === "text") {
						currentBlock.text += choice.delta.content;
						stream.push({
							type: "text_delta",
							contentIndex: blockIndex(),
							delta: choice.delta.content,
							partial: output,
						});
					}
				}

				// Handle reasoning content (reasoning_content, reasoning, reasoning_text)
				const reasoningFields = ["reasoning_content", "reasoning", "reasoning_text"];
				let foundReasoningField: string | null = null;
				for (const field of reasoningFields) {
					const value = (choice.delta as any)[field];
					if (value !== null && value !== undefined && value.length > 0) {
						foundReasoningField = field;
						break;
					}
				}

				if (foundReasoningField) {
					if (!currentBlock || currentBlock.type !== "thinking") {
						finishCurrentBlock(currentBlock);
						currentBlock = {
							type: "thinking",
							thinking: "",
							thinkingSignature: foundReasoningField,
						};
						output.content.push(currentBlock);
						stream.push({ type: "thinking_start", contentIndex: blockIndex(), partial: output });
					}

					if (currentBlock.type === "thinking") {
						const delta = (choice.delta as any)[foundReasoningField];
						currentBlock.thinking += delta;
						stream.push({
							type: "thinking_delta",
							contentIndex: blockIndex(),
							delta,
							partial: output,
						});
					}
				}

				// Handle MiniMax reasoning_details format
				const reasoningDetails = (choice.delta as any).reasoning_details;
				if (reasoningDetails && Array.isArray(reasoningDetails)) {
					for (const detail of reasoningDetails) {
						if (detail.type === "reasoning.text" && detail.text) {
							if (!currentBlock || currentBlock.type !== "thinking") {
								finishCurrentBlock(currentBlock);
								currentBlock = {
									type: "thinking",
									thinking: "",
									thinkingSignature: "reasoning_details",
								};
								output.content.push(currentBlock);
								stream.push({ type: "thinking_start", contentIndex: blockIndex(), partial: output });
							}

							if (currentBlock.type === "thinking") {
								currentBlock.thinking += detail.text;
								stream.push({
									type: "thinking_delta",
									contentIndex: blockIndex(),
									delta: detail.text,
									partial: output,
								});
							}
						}
					}
				}

				// Handle tool calls
				if (choice.delta.tool_calls) {
					for (const toolCall of choice.delta.tool_calls) {
						if (
							!currentBlock ||
							currentBlock.type !== "toolCall" ||
							(toolCall.id && currentBlock.id !== toolCall.id)
						) {
							finishCurrentBlock(currentBlock);
							currentBlock = {
								type: "toolCall",
								id: toolCall.id || "",
								name: toolCall.function?.name || "",
								arguments: {},
								partialArgs: "",
							};
							output.content.push(currentBlock);
							stream.push({ type: "toolcall_start", contentIndex: blockIndex(), partial: output });
						}

						if (currentBlock.type === "toolCall") {
							if (toolCall.id) currentBlock.id = toolCall.id;
							if (toolCall.function?.name) currentBlock.name = toolCall.function.name;
							let delta = "";
							if (toolCall.function?.arguments) {
								delta = toolCall.function.arguments;
								currentBlock.partialArgs += toolCall.function.arguments;
								currentBlock.arguments = parseStreamingJson(currentBlock.partialArgs);
							}
							stream.push({
								type: "toolcall_delta",
								contentIndex: blockIndex(),
								delta,
								partial: output,
							});
						}
					}
				}
			}

			finishCurrentBlock(currentBlock);

			if (options?.signal?.aborted) {
				throw new Error("Request was aborted");
			}

			if (output.stopReason === "aborted" || output.stopReason === "error") {
				throw new Error("An unknown error occurred");
			}

			stream.push({ type: "done", reason: output.stopReason, message: output });
			stream.end();
		} catch (error) {
			for (const block of output.content) delete (block as any).index;
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
};

export const streamSimpleIflow: StreamFunction<"iflow-completions", SimpleStreamOptions> = (
	model: Model<"iflow-completions">,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream => {
	const apiKey = options?.apiKey || getEnvApiKey(model.provider);
	if (!apiKey) {
		throw new Error(`No API key for provider: ${model.provider}`);
	}

	const base = buildBaseOptions(model, options, apiKey);
	const toolChoice = (options as IflowOptions | undefined)?.toolChoice;
	const modelSupportsSetThink = supportsSetThink(model.id);

	const reasoningEffort = modelSupportsSetThink && options?.reasoning ? clampReasoning(options.reasoning) : undefined;
	const thinking = modelSupportsSetThink
		? options?.reasoning
			? { type: "enabled" as const }
			: { type: "disabled" as const }
		: undefined;

	return streamIflow(model, context, {
		...base,
		reasoningEffort,
		toolChoice,
		thinking,
	} as unknown as IflowOptions);
};
