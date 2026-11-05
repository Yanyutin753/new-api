package oaichat

import (
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/convmeta"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResponseOpenAI2GeminiMapsTextToolFinishReasonAndUsage(t *testing.T) {
	msg := dto.Message{
		Role:    "assistant",
		Content: "hello",
	}
	msg.SetToolCalls([]dto.ToolCallRequest{
		{
			ID:   "call_1",
			Type: "function",
			Function: dto.FunctionRequest{
				Name:      "lookup",
				Arguments: `{"q":"x"}`,
			},
		},
	})

	resp := ResponseOpenAI2Gemini(&dto.OpenAITextResponse{
		Model: "gpt-test",
		Choices: []dto.OpenAITextResponseChoice{
			{
				Index:        2,
				Message:      msg,
				FinishReason: "length",
			},
		},
		Usage: dto.Usage{
			PromptTokens:     11,
			CompletionTokens: 5,
			TotalTokens:      16,
		},
	}, nil)

	assert.Equal(t, 11, resp.UsageMetadata.PromptTokenCount)
	assert.Equal(t, 5, resp.UsageMetadata.CandidatesTokenCount)
	assert.Equal(t, 16, resp.UsageMetadata.TotalTokenCount)
	require.NotNil(t, resp.UsageMetadata.BillingUsage)
	require.NotNil(t, resp.UsageMetadata.BillingUsage.OpenAIUsage)
	assert.Equal(t, dto.BillingUsageSourceOAIChat, resp.UsageMetadata.BillingUsage.Source)
	assert.Equal(t, dto.BillingUsageSemanticOpenAI, resp.UsageMetadata.BillingUsage.Semantic)
	assert.Equal(t, 11, resp.UsageMetadata.BillingUsage.OpenAIUsage.PromptTokens)
	assert.Equal(t, 5, resp.UsageMetadata.BillingUsage.OpenAIUsage.CompletionTokens)
	assert.Equal(t, 16, resp.UsageMetadata.BillingUsage.OpenAIUsage.TotalTokens)
	assert.Nil(t, resp.UsageMetadata.BillingUsage.OpenAIUsage.BillingUsage)
	require.Len(t, resp.Candidates, 1)
	assert.Equal(t, int64(2), resp.Candidates[0].Index)
	require.NotNil(t, resp.Candidates[0].FinishReason)
	assert.Equal(t, "MAX_TOKENS", *resp.Candidates[0].FinishReason)
	require.Len(t, resp.Candidates[0].Content.Parts, 2)
	assert.Equal(t, "hello", resp.Candidates[0].Content.Parts[0].Text)
	require.NotNil(t, resp.Candidates[0].Content.Parts[1].FunctionCall)
	assert.Equal(t, "lookup", resp.Candidates[0].Content.Parts[1].FunctionCall.FunctionName)
	assert.Equal(t, map[string]any{"q": "x"}, resp.Candidates[0].Content.Parts[1].FunctionCall.Arguments)
}

func TestStreamResponseOpenAI2GeminiMapsToolCallFinishReasonAndUsage(t *testing.T) {
	resp := StreamResponseOpenAI2Gemini(&dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{
				Index:        1,
				FinishReason: geminiRespPtr("tool_calls"),
				Delta: dto.ChatCompletionsStreamResponseChoiceDelta{
					ToolCalls: []dto.ToolCallResponse{
						{
							Type: "function",
							Function: dto.FunctionResponse{
								Name:      "lookup",
								Arguments: `{"q":"x"}`,
							},
						},
					},
				},
			},
		},
		Usage: &dto.Usage{
			PromptTokens:     13,
			CompletionTokens: 8,
			TotalTokens:      21,
		},
	}, &convmeta.Values{})

	require.NotNil(t, resp)
	assert.Equal(t, 13, resp.UsageMetadata.PromptTokenCount)
	assert.Equal(t, 8, resp.UsageMetadata.CandidatesTokenCount)
	assert.Equal(t, 21, resp.UsageMetadata.TotalTokenCount)
	require.NotNil(t, resp.UsageMetadata.BillingUsage)
	require.NotNil(t, resp.UsageMetadata.BillingUsage.OpenAIUsage)
	assert.Equal(t, 13, resp.UsageMetadata.BillingUsage.OpenAIUsage.PromptTokens)
	assert.Equal(t, 8, resp.UsageMetadata.BillingUsage.OpenAIUsage.CompletionTokens)
	require.Len(t, resp.Candidates, 1)
	assert.Equal(t, int64(1), resp.Candidates[0].Index)
	require.NotNil(t, resp.Candidates[0].FinishReason)
	assert.Equal(t, "STOP", *resp.Candidates[0].FinishReason)
	require.Len(t, resp.Candidates[0].Content.Parts, 1)
	require.NotNil(t, resp.Candidates[0].Content.Parts[0].FunctionCall)
	assert.Equal(t, "lookup", resp.Candidates[0].Content.Parts[0].FunctionCall.FunctionName)
	assert.Equal(t, map[string]any{"q": "x"}, resp.Candidates[0].Content.Parts[0].FunctionCall.Arguments)
}

func geminiRespPtr[T any](value T) *T {
	return &value
}

func TestChatToGeminiPreservesCachedTokensAcrossResponsePaths(t *testing.T) {
	for _, cached := range []int{0, 700} {
		usage := dto.Usage{PromptTokens: 1000, CompletionTokens: 50, TotalTokens: 1050}
		usage.PromptTokensDetails.CachedTokens = cached
		finish := dto.ChatCompletionsStreamResponseChoice{FinishReason: geminiRespPtr("stop")}
		for name, convert := range map[string]func(*testing.T, *dto.Usage) *dto.GeminiChatResponse{
			"non-stream": func(t *testing.T, u *dto.Usage) *dto.GeminiChatResponse {
				return ResponseOpenAI2Gemini(&dto.OpenAITextResponse{Usage: *u}, nil)
			},
			"legacy-stream": func(t *testing.T, u *dto.Usage) *dto.GeminiChatResponse {
				return StreamResponseOpenAI2Gemini(&dto.ChatCompletionsStreamResponse{
					Choices: []dto.ChatCompletionsStreamResponseChoice{finish}, Usage: u,
				}, nil)
			},
			"usage-only-final-chunk": func(t *testing.T, u *dto.Usage) *dto.GeminiChatResponse {
				state := NewChatToGeminiStreamState()
				_, err := state.ConvertChunk(&dto.ChatCompletionsStreamResponse{
					Choices: []dto.ChatCompletionsStreamResponseChoice{finish},
				}, nil)
				require.NoError(t, err)
				responses, err := state.ConvertChunk(&dto.ChatCompletionsStreamResponse{Usage: u}, nil)
				require.NoError(t, err)
				require.Len(t, responses, 1)
				final, err := state.Finalize(nil)
				require.NoError(t, err)
				assert.Empty(t, final)
				return responses[0]
			},
			"finalize": func(t *testing.T, u *dto.Usage) *dto.GeminiChatResponse {
				state := NewChatToGeminiStreamState()
				state.SetUsage(u)
				responses, err := state.Finalize(nil)
				require.NoError(t, err)
				require.Len(t, responses, 1)
				return responses[0]
			},
		} {
			t.Run(name, func(t *testing.T) {
				response := convert(t, &usage)
				require.NotNil(t, response)
				assert.Equal(t, cached, response.UsageMetadata.CachedContentTokenCount)
				assert.Equal(t, 1000, response.UsageMetadata.PromptTokenCount)
				assert.Equal(t, 50, response.UsageMetadata.CandidatesTokenCount)
				assert.Equal(t, 1050, response.UsageMetadata.TotalTokenCount)
				require.NotNil(t, response.UsageMetadata.BillingUsage)
				require.NotNil(t, response.UsageMetadata.BillingUsage.OpenAIUsage)
				assert.Equal(t, cached, response.UsageMetadata.BillingUsage.OpenAIUsage.PromptTokensDetails.CachedTokens)
				encoded, err := kitutil.Marshal(response)
				require.NoError(t, err)
				var wire struct {
					UsageMetadata struct {
						Cached *int `json:"cachedContentTokenCount"`
					} `json:"usageMetadata"`
				}
				require.NoError(t, kitutil.Unmarshal(encoded, &wire))
				require.NotNil(t, wire.UsageMetadata.Cached)
				assert.Equal(t, cached, *wire.UsageMetadata.Cached)
			})
		}
	}
}
