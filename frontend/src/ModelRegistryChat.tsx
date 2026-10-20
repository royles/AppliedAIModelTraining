import { useCallback, useRef, useState } from "react";
import type { RegistryModel } from "./api";
import { chatWithModel, streamChatWithModel, type ChatMessage } from "./chatApi";

const SERVING = new Set(["trained", "testing", "benchmarked", "production"]);

type Props = {
  model: RegistryModel;
};

type UiMessage = ChatMessage & { id: string };

function newId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function ModelRegistryChat({ model }: Props) {
  const [systemPrompt, setSystemPrompt] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [streaming, setStreaming] = useState(true);
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const canChat = model.task_type === "causal_lm" && SERVING.has(model.lifecycle_state);

  const buildPayloadMessages = useCallback((): ChatMessage[] => {
    const convo: ChatMessage[] = messages.map(({ role, content }) => ({ role, content }));
    if (systemPrompt.trim()) {
      return [{ role: "system", content: systemPrompt.trim() }, ...convo];
    }
    return convo;
  }, [messages, systemPrompt]);

  async function send() {
    const text = input.trim();
    if (!text || busy || !canChat) return;
    setError(null);
    setInput("");
    const userMsg: UiMessage = { id: newId(), role: "user", content: text };
    const assistantId = newId();
    setMessages((prev) => [...prev, userMsg, { id: assistantId, role: "assistant", content: "" }]);
    setBusy(true);

    const baseMessages = [...buildPayloadMessages(), { role: "user", content: text }];

    try {
      if (streaming) {
        await streamChatWithModel(
          {
            model: model.openai_model_id,
            messages: baseMessages,
            max_tokens: maxTokens,
            temperature,
          },
          (delta) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: m.content + delta } : m,
              ),
            );
          },
        );
      } else {
        const reply = await chatWithModel({
          model: model.openai_model_id,
          messages: baseMessages,
          max_tokens: maxTokens,
          temperature,
          stream: false,
        });
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: reply } : m)),
        );
      }
    } catch (e) {
      setMessages((prev) => prev.filter((m) => m.id !== assistantId));
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }

  function clearChat() {
    setMessages([]);
    setError(null);
  }

  if (model.task_type !== "causal_lm") {
    return (
      <p className="hint registry-chat-disabled">
        Chat is only available for causal language models.
      </p>
    );
  }

  if (!SERVING.has(model.lifecycle_state)) {
    return (
      <p className="hint registry-chat-disabled">
        Model is in state <strong>{model.lifecycle_state}</strong>. Promote to trained or later
        to enable chat (same rules as <code>/v1/chat/completions</code>).
      </p>
    );
  }

  return (
    <section className="registry-chat" aria-label="Model chat">
      <h4>Chat with this model</h4>
      <p className="hint">
        Uses <code>POST /v1/chat/completions</code> against <code>{model.openai_model_id}</code>.
      </p>

      <label className="registry-chat-field">
        <span>System prompt override</span>
        <textarea
          rows={3}
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="Optional — prepended as a system message for each request"
          disabled={busy}
        />
      </label>

      <div className="registry-chat-options">
        <label>
          <input
            type="checkbox"
            checked={streaming}
            onChange={(e) => setStreaming(e.target.checked)}
            disabled={busy}
          />
          Stream responses
        </label>
        <label>
          Max tokens
          <input
            type="number"
            min={1}
            max={4096}
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value) || 256)}
            disabled={busy}
          />
        </label>
        <label>
          Temperature
          <input
            type="number"
            min={0}
            max={2}
            step={0.1}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            disabled={busy}
          />
        </label>
        <button type="button" className="btn-secondary" onClick={clearChat} disabled={busy}>
          Clear chat
        </button>
      </div>

      {error && <div className="banner error registry-chat-error">{error}</div>}

      <div className="registry-chat-log" ref={scrollRef}>
        {messages.length === 0 ? (
          <p className="hint">Send a message to try the registered model.</p>
        ) : (
          messages.map((m) => (
            <div key={m.id} className={`registry-chat-bubble registry-chat-${m.role}`}>
              <span className="registry-chat-role">{m.role}</span>
              <div className="registry-chat-content">{m.content || (busy ? "…" : "")}</div>
            </div>
          ))
        )}
      </div>

      <form
        className="registry-chat-compose"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <textarea
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="User message…"
          disabled={busy || !canChat}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button type="submit" className="btn-primary" disabled={busy || !input.trim()}>
          {busy ? "Generating…" : "Send"}
        </button>
      </form>
    </section>
  );
}
