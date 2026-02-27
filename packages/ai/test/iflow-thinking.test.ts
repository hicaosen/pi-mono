import { Type } from "@sinclair/typebox";
import { describe, expect, it } from "vitest";
import { applyModelSpecificParams, convertIflowMessages } from "../src/providers/iflow.js";
import type { AssistantMessage, Context, Model, Tool, UserMessage } from "../src/types.js";

function makeIflowKimiK2ThinkingModel(): Model<"iflow-completions"> {
	return {
		id: "kimi-k2-thinking",
		name: "Kimi K2 Thinking",
		api: "iflow-completions",
		provider: "iflow",
		baseUrl: "https://apis.iflow.cn/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 16000,
	};
}

function makeIflowKimiK2_5Model(): Model<"iflow-completions"> {
	return {
		id: "kimi-k2.5",
		name: "Kimi K2.5",
		api: "iflow-completions",
		provider: "iflow",
		baseUrl: "https://apis.iflow.cn/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 16000,
	};
}

function makeIflowGLM47Model(): Model<"iflow-completions"> {
	return {
		id: "glm-4.7",
		name: "GLM-4.7",
		api: "iflow-completions",
		provider: "iflow",
		baseUrl: "https://apis.iflow.cn/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens: 32000,
	};
}

function makeIflowGLM5Model(): Model<"iflow-completions"> {
	return {
		id: "glm-5",
		name: "GLM-5",
		api: "iflow-completions",
		provider: "iflow",
		baseUrl: "https://apis.iflow.cn/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 16000,
	};
}

function makeIflowDeepSeekV32Model(): Model<"iflow-completions"> {
	return {
		id: "deepseek-v3.2",
		name: "DeepSeek-V3.2",
		api: "iflow-completions",
		provider: "iflow",
		baseUrl: "https://apis.iflow.cn/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 16000,
	};
}

function makeIflowRomeModel(): Model<"iflow-completions"> {
	return {
		id: "iFlow-ROME-30BA3B",
		name: "iFlow-ROME-30BA3B",
		api: "iflow-completions",
		provider: "iflow",
		baseUrl: "https://apis.iflow.cn/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 256000,
		maxTokens: 64000,
	};
}

function makeIflowMiniMaxM25Model(): Model<"iflow-completions"> {
	return {
		id: "minimax-m2.5",
		name: "MiniMax-M2.5",
		api: "iflow-completions",
		provider: "iflow",
		baseUrl: "https://apis.iflow.cn/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 64000,
	};
}

function makeIflowKimiK20905Model(): Model<"iflow-completions"> {
	return {
		id: "kimi-k2-0905",
		name: "Kimi-K2-0905",
		api: "iflow-completions",
		provider: "iflow",
		baseUrl: "https://apis.iflow.cn/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 256000,
		maxTokens: 32000,
	};
}

describe("iFlow thinking model support", () => {
	describe("convertIflowMessages for Kimi K2 models", () => {
		it("should convert assistant message with thinking content to reasoning_content field for kimi-k2-thinking", () => {
			const model = makeIflowKimiK2ThinkingModel();

			const previousAssistantMsg: AssistantMessage = {
				role: "assistant",
				content: [
					{
						type: "thinking",
						thinking: "Let me analyze this problem step by step...",
						thinkingSignature: "reasoning_content",
					},
					{
						type: "text",
						text: "The answer is 42.",
					},
				],
				api: "iflow-completions",
				provider: "iflow",
				model: "kimi-k2-thinking",
				usage: {
					input: 10,
					output: 20,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 30,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};

			const userMsg: UserMessage = {
				role: "user",
				content: "Can you explain why?",
				timestamp: Date.now(),
			};

			const context: Context = {
				messages: [previousAssistantMsg, userMsg],
			};

			const result = convertIflowMessages(model, context);

			// Find the assistant message in the result
			const assistantMsg = result.find((m) => m.role === "assistant") as Record<string, unknown> | undefined;
			expect(assistantMsg).toBeDefined();
			expect(assistantMsg?.reasoning_content).toBe("Let me analyze this problem step by step...");
			expect(assistantMsg?.content).toEqual([{ type: "text", text: "The answer is 42." }]);
		});

		it("should convert assistant message with thinking content to reasoning_content field for kimi-k2.5", () => {
			const model = makeIflowKimiK2_5Model();

			const previousAssistantMsg: AssistantMessage = {
				role: "assistant",
				content: [
					{
						type: "thinking",
						thinking: "Let me think about this carefully...",
						thinkingSignature: "reasoning_content",
					},
					{
						type: "text",
						text: "Based on my analysis, the solution is...",
					},
				],
				api: "iflow-completions",
				provider: "iflow",
				model: "kimi-k2.5",
				usage: {
					input: 15,
					output: 25,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 40,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};

			const context: Context = {
				messages: [previousAssistantMsg],
			};

			const result = convertIflowMessages(model, context);

			const assistantMsg = result.find((m) => m.role === "assistant") as Record<string, unknown> | undefined;
			expect(assistantMsg).toBeDefined();
			expect(assistantMsg?.reasoning_content).toBe("Let me think about this carefully...");
			expect(assistantMsg?.content).toEqual([{ type: "text", text: "Based on my analysis, the solution is..." }]);
		});

		it("should convert assistant message with multiple thinking blocks", () => {
			const model = makeIflowKimiK2_5Model();

			const previousAssistantMsg: AssistantMessage = {
				role: "assistant",
				content: [
					{
						type: "thinking",
						thinking: "First, let me understand the problem...",
						thinkingSignature: "reasoning_content",
					},
					{
						type: "thinking",
						thinking: "Then, I need to consider the edge cases...",
						thinkingSignature: "reasoning_content",
					},
					{
						type: "text",
						text: "The solution is to use dynamic programming.",
					},
				],
				api: "iflow-completions",
				provider: "iflow",
				model: "kimi-k2.5",
				usage: {
					input: 15,
					output: 25,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 40,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};

			const context: Context = {
				messages: [previousAssistantMsg],
			};

			const result = convertIflowMessages(model, context);

			const assistantMsg = result.find((m) => m.role === "assistant") as Record<string, unknown> | undefined;
			expect(assistantMsg).toBeDefined();
			// Multiple thinking blocks should be joined with newlines
			expect(assistantMsg?.reasoning_content).toBe(
				"First, let me understand the problem...\nThen, I need to consider the edge cases...",
			);
		});

		it("should handle assistant message without thinking content", () => {
			const model = makeIflowKimiK2ThinkingModel();

			const previousAssistantMsg: AssistantMessage = {
				role: "assistant",
				content: [
					{
						type: "text",
						text: "Hello! How can I help you today?",
					},
				],
				api: "iflow-completions",
				provider: "iflow",
				model: "kimi-k2-thinking",
				usage: {
					input: 5,
					output: 10,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 15,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};

			const context: Context = {
				messages: [previousAssistantMsg],
			};

			const result = convertIflowMessages(model, context);

			const assistantMsg = result.find((m) => m.role === "assistant") as Record<string, unknown> | undefined;
			expect(assistantMsg).toBeDefined();
			expect(assistantMsg?.reasoning_content).toBeUndefined();
			expect(assistantMsg?.content).toEqual([{ type: "text", text: "Hello! How can I help you today?" }]);
		});
	});

	describe("convertIflowMessages for reasoning models", () => {
		it("should add reasoning_content field for GLM-5 model (reasoning=true)", () => {
			const model = makeIflowGLM5Model();

			const previousAssistantMsg: AssistantMessage = {
				role: "assistant",
				content: [
					{
						type: "thinking",
						thinking: "Let me analyze this...",
						thinkingSignature: "reasoning_content",
					},
					{
						type: "text",
						text: "The answer is 42.",
					},
				],
				api: "iflow-completions",
				provider: "iflow",
				model: "glm-5",
				usage: {
					input: 10,
					output: 20,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 30,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};

			const context: Context = {
				messages: [previousAssistantMsg],
			};

			const result = convertIflowMessages(model, context);

			const assistantMsg = result.find((m) => m.role === "assistant") as Record<string, unknown> | undefined;
			expect(assistantMsg).toBeDefined();
			// All reasoning=true models should have reasoning_content field
			expect(assistantMsg?.reasoning_content).toBe("Let me analyze this...");
			expect(assistantMsg?.content).toEqual([{ type: "text", text: "The answer is 42." }]);
		});

		it("should add reasoning_content field for DeepSeek-V3.2 model (reasoning=true)", () => {
			const model = makeIflowDeepSeekV32Model();

			const previousAssistantMsg: AssistantMessage = {
				role: "assistant",
				content: [
					{
						type: "thinking",
						thinking: "Let me analyze this...",
						thinkingSignature: "reasoning_content",
					},
					{
						type: "text",
						text: "The answer is 42.",
					},
				],
				api: "iflow-completions",
				provider: "iflow",
				model: "deepseek-v3.2",
				usage: {
					input: 10,
					output: 20,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 30,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};

			const context: Context = {
				messages: [previousAssistantMsg],
			};

			const result = convertIflowMessages(model, context);

			const assistantMsg = result.find((m) => m.role === "assistant") as Record<string, unknown> | undefined;
			expect(assistantMsg).toBeDefined();
			// All reasoning=true models should have reasoning_content field
			expect(assistantMsg?.reasoning_content).toBe("Let me analyze this...");
			expect(assistantMsg?.content).toEqual([{ type: "text", text: "The answer is 42." }]);
		});

		it("should preserve reasoning_content in single-turn conversation", () => {
			const model = makeIflowDeepSeekV32Model();

			const assistantMsg: AssistantMessage = {
				role: "assistant",
				content: [
					{
						type: "thinking",
						thinking: "Single turn reasoning...",
						thinkingSignature: "reasoning_content",
					},
					{
						type: "text",
						text: "Single turn answer.",
					},
				],
				api: "iflow-completions",
				provider: "iflow",
				model: "deepseek-v3.2",
				usage: {
					input: 10,
					output: 20,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 30,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};

			const userMsg: UserMessage = {
				role: "user",
				content: "Follow up",
				timestamp: Date.now(),
			};

			// Single turn: assistant followed by user (no previous user message)
			const context: Context = {
				messages: [assistantMsg, userMsg],
			};

			const result = convertIflowMessages(model, context);

			// Should have assistant, user
			expect(result).toHaveLength(2);

			// reasoning_content should be preserved in single-turn conversation
			const resultAssistant = result.find((m) => m.role === "assistant") as Record<string, unknown> | undefined;
			expect(resultAssistant).toBeDefined();
			expect(resultAssistant?.reasoning_content).toBe("Single turn reasoning...");
		});
	});

	describe("common functionality", () => {
		it("should include system prompt when provided", () => {
			const model = makeIflowKimiK2_5Model();

			const context: Context = {
				systemPrompt: "You are a helpful assistant.",
				messages: [
					{
						role: "user",
						content: "Hello",
						timestamp: Date.now(),
					},
				],
			};

			const result = convertIflowMessages(model, context);

			expect(result[0]).toEqual({
				role: "system",
				content: "You are a helpful assistant.",
			});
		});

		it("should handle empty thinking blocks for Kimi models", () => {
			const model = makeIflowKimiK2ThinkingModel();

			const previousAssistantMsg: AssistantMessage = {
				role: "assistant",
				content: [
					{
						type: "thinking",
						thinking: "   ", // Empty/whitespace thinking
						thinkingSignature: "reasoning_content",
					},
					{
						type: "text",
						text: "Some response",
					},
				],
				api: "iflow-completions",
				provider: "iflow",
				model: "kimi-k2-thinking",
				usage: {
					input: 5,
					output: 5,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 10,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};

			const context: Context = {
				messages: [previousAssistantMsg],
			};

			const result = convertIflowMessages(model, context);

			const assistantMsg = result.find((m) => m.role === "assistant") as Record<string, unknown> | undefined;
			expect(assistantMsg).toBeDefined();
			// Empty thinking blocks should be filtered out
			expect(assistantMsg?.reasoning_content).toBeUndefined();
			expect(assistantMsg?.content).toEqual([{ type: "text", text: "Some response" }]);
		});
	});

	describe("applyModelSpecificParams for glm-4.7", () => {
		it("should set glm-4.7 defaults and enable_thinking=true when thinking is enabled", () => {
			const model = makeIflowGLM47Model();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, { thinking: { type: "enabled" } }, context);

			expect(params.temperature).toBe(1);
			expect(params.top_p).toBe(0.95);
			expect(params.chat_template_kwargs).toEqual({ enable_thinking: true });
		});

		it("should set enable_thinking=false when thinking is disabled", () => {
			const model = makeIflowGLM47Model();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, { thinking: { type: "disabled" } }, context);

			expect(params.chat_template_kwargs).toEqual({ enable_thinking: false });
		});
	});

	describe("applyModelSpecificParams for deepseek-v3.2", () => {
		it("should rewrite model and enable thinking_mode when thinking is enabled", () => {
			const model = makeIflowDeepSeekV32Model();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, { reasoning: "high" }, context);

			expect(params.model).toBe("deepseek-v3.2-reasoner");
			expect(params.thinking_mode).toBe(true);
			expect(params.reasoning).toBe(true);
		});

		it("should not set reasoning=true when level is low", () => {
			const model = makeIflowDeepSeekV32Model();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, { reasoning: "low" }, context);

			expect(params.model).toBe("deepseek-v3.2-reasoner");
			expect(params.thinking_mode).toBe(true);
			expect(params.reasoning).toBeUndefined();
		});
	});

	describe("applyModelSpecificParams for glm-5", () => {
		it("should set glm-5 thinking params when enabled", () => {
			const model = makeIflowGLM5Model();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, { thinking: { type: "enabled" } }, context);

			expect(params.temperature).toBe(1);
			expect(params.top_p).toBe(0.95);
			expect(params.chat_template_kwargs).toEqual({ enable_thinking: true });
			expect(params.enable_thinking).toBe(true);
			expect(params.thinking).toEqual({ type: "enabled" });
		});

		it("should set glm-5 thinking params when disabled", () => {
			const model = makeIflowGLM5Model();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, { thinking: { type: "disabled" } }, context);

			expect(params.chat_template_kwargs).toEqual({ enable_thinking: false });
			expect(params.enable_thinking).toBe(false);
			expect(params.thinking).toEqual({ type: "disabled" });
		});
	});

	describe("applyModelSpecificParams for iFlow-ROME-30BA3B", () => {
		it("should enforce temperature/top_p/top_k", () => {
			const model = makeIflowRomeModel();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, undefined, context);

			expect(params.temperature).toBe(0.7);
			expect(params.top_p).toBe(0.8);
			expect(params.top_k).toBe(20);
		});
	});

	describe("applyModelSpecificParams for kimi-k2-thinking", () => {
		it("should set thinking_mode=true when thinking is enabled", () => {
			const model = makeIflowKimiK2ThinkingModel();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, { thinking: { type: "enabled" } }, context);

			expect(params.thinking_mode).toBe(true);
			expect(params.thinking).toBeUndefined();
		});

		it("should not set thinking parameters when disabled", () => {
			const model = makeIflowKimiK2ThinkingModel();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, { thinking: { type: "disabled" } }, context);

			expect(params.thinking_mode).toBeUndefined();
			expect(params.thinking).toBeUndefined();
		});
	});

	describe("applyModelSpecificParams for kimi-k2.5", () => {
		it("should enforce kimi-k2.5 required params when thinking is enabled", () => {
			const model = makeIflowKimiK2_5Model();
			const context: Context = { messages: [] };
			const options = { thinking: { type: "enabled" as const } };
			const params = applyModelSpecificParams(model, options, context);

			expect(params.temperature).toBeUndefined();
			expect(params.top_p).toBe(0.95);
			expect(params.n).toBe(1);
			expect(params.presence_penalty).toBe(0);
			expect(params.frequency_penalty).toBe(0);
			expect(params.max_tokens).toBe(model.maxTokens);
			expect(params.thinking).toEqual({ type: "enabled" });
		});

		it("should enforce kimi-k2.5 required params when thinking is disabled", () => {
			const model = makeIflowKimiK2_5Model();
			const context: Context = { messages: [] };
			const options = { thinking: { type: "disabled" as const } };
			const params = applyModelSpecificParams(model, options, context);

			expect(params.temperature).toBeUndefined();
			expect(params.thinking).toEqual({ type: "disabled" });
		});

		it("should force tool_choice to auto when thinking is enabled and invalid toolChoice is provided", () => {
			const model = makeIflowKimiK2_5Model();
			const tool: Tool = {
				name: "test_tool",
				description: "Test tool",
				parameters: Type.Object({}),
			};
			const context: Context = { messages: [], tools: [tool] };
			const options = {
				thinking: { type: "enabled" as const },
				toolChoice: "required" as const,
			};

			const params = applyModelSpecificParams(model, options, context);
			expect(params.tool_choice).toBe("auto");
		});
	});

	describe("applyModelSpecificParams for unsupported setThink models", () => {
		it("should not add thinking fields for minimax-m2.5", () => {
			const model = makeIflowMiniMaxM25Model();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, { reasoning: "high" }, context);

			expect(params.reasoning_split).toBeUndefined();
			expect(params.thinking).toBeUndefined();
			expect(params.thinking_mode).toBeUndefined();
		});

		it("should not add thinking fields for kimi-k2-0905", () => {
			const model = makeIflowKimiK20905Model();
			const context: Context = { messages: [] };
			const params = applyModelSpecificParams(model, { reasoning: "high" }, context);

			expect(params.thinking).toBeUndefined();
			expect(params.thinking_mode).toBeUndefined();
			expect(params.enable_thinking).toBeUndefined();
		});
	});
});
