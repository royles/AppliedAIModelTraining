import { TransformerArchitectureDiagram } from "../TransformerArchitectureDiagram";

type Props = {
  presetTitle?: string;
  family: "gpt2" | "bert";
  initMode: "from_scratch" | "pretrained";
  taskType: "causal_lm" | "sequence_classification";
  baseModel?: string | null;
  useLora?: boolean;
  hiddenSize: number;
  layers: number;
  heads: number;
  vocabSize: number;
  maxPosition: number;
  intermediateSize: number;
  tieWordEmbeddings: boolean;
  paramEstimate: number | null;
};

export function DiagramWizardStep(props: Props) {
  return (
    <div className="wizard-step-compact wizard-step-diagram">
      <TransformerArchitectureDiagram {...props} compact />
    </div>
  );
}
