import { resolveApiUrl } from "./api";

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type ChatCompletionBody = {
  model: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
};

type ChatCompletionResponse = {
  choices: { message: { content: string } }[];
};

async function parseError(res: Response, fallback: string): Promise<string> {
  const text = await res.text();
  try {
    const body = JSON.parse(text) as { detail?: string };
    if (body.detail) return body.detail;
  } catch {
    /* use raw */
  }
  return text || fallback;
}

export async function chatWithModel(body: ChatCompletionBody): Promise<string> {
  const res = await fetch(resolveApiUrl("/v1/chat/completions"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: false }),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, res.statusText));
  }
  const data = (await res.json()) as ChatCompletionResponse;
  return data.choices[0]?.message?.content ?? "";
}

export async function streamChatWithModel(
  body: ChatCompletionBody,
  onDelta: (text: string) => void,
): Promise<void> {
  const res = await fetch(resolveApiUrl("/v1/chat/completions"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: true }),
  });
  if (!res.ok) {
    throw new Error(await parseError(res, res.statusText));
  }
  const reader = res.body?.getReader();
  if (!reader) {
    throw new Error("Streaming not supported in this browser");
  }
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n");
    buffer = parts.pop() ?? "";
    for (const line of parts) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (data === "[DONE]") return;
      try {
        const chunk = JSON.parse(data) as {
          error?: { message?: string };
          choices?: { delta?: { content?: string } }[];
        };
        if (chunk.error?.message) {
          throw new Error(chunk.error.message);
        }
        const delta = chunk.choices?.[0]?.delta?.content;
        if (delta) onDelta(delta);
      } catch (e) {
        if (e instanceof Error && e.message !== "Unexpected end of JSON input") {
          throw e;
        }
      }
    }
  }
}
