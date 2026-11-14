from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .inference import ModelSpec, ThinkingMode


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationCreate(StrictModel):
    title: str = Field(default="New conversation", min_length=1, max_length=200)
    logging_enabled: bool = True
    training_enabled: bool = False

    @model_validator(mode="after")
    def consent(self):
        if self.training_enabled and not self.logging_enabled:
            raise ValueError("Enable logging before opting into training")
        return self


class ConversationPatch(StrictModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    logging_enabled: bool | None = None
    training_enabled: bool | None = None


class KnowledgeAttachment(StrictModel):
    filename: str | None = Field(default=None, min_length=1, max_length=200)
    content_base64: str | None = Field(default=None, max_length=178956972)
    path: str | None = Field(default=None, min_length=1, max_length=4096)
    url: str | None = Field(default=None, min_length=1, max_length=8192)
    video_frames: int = Field(default=8, ge=1, le=24)

    @model_validator(mode="after")
    def one_source(self):
        if sum(value is not None for value in (self.content_base64, self.path, self.url)) != 1:
            raise ValueError("Provide exactly one uploaded file, local path or public URL")
        return self


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    # Character counts do not represent the selected model's token context.
    # The inference engine validates its actual prompt plus generation budget.
    content: str = Field(min_length=1)


class ChatRequest(StrictModel):
    conversation_id: str | None = None
    model: str | None = None
    messages: list[Message] = Field(min_length=1, max_length=200)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1)
    thinking: ThinkingMode | None = None
    use_memory: bool = False

    @model_validator(mode="after")
    def last_user(self):
        if self.messages[-1].role != "user":
            raise ValueError("The last message must be from the user")
        return self


class FeedbackRequest(StrictModel):
    conversation_id: str
    message_id: str
    label: Literal["good", "bad", "correct", "too_verbose", "too_short", "wrong_fact", "unsafe", "refusal", "format_error", "tool_error", "user_edited", "user_liked", "user_disliked"]
    reason: str | None = Field(default=None, max_length=2000)
    correction: str | None = Field(default=None, max_length=100000)

    @model_validator(mode="after")
    def valid_correction(self):
        if self.label in {"correct", "user_edited"} and not (self.correction or "").strip():
            raise ValueError("A correction requires replacement answer text")
        return self


class ModelLoad(StrictModel):
    model_id: str | None = None
    spec: ModelSpec | None = None

    @model_validator(mode="after")
    def provided_model(self):
        if self.spec is None and not self.model_id:
            raise ValueError("Provide a registered model_id or an explicit model spec")
        if self.spec is not None:
            if self.model_id is not None and self.model_id != self.spec.id:
                raise ValueError("model_id must match spec.id")
            self.model_id = self.spec.id
        return self


class ModelUnload(StrictModel):
    model_id: str | None = None


class AdapterRequest(StrictModel):
    adapter_id: str


class AdapterOptional(StrictModel):
    adapter_id: str | None = None


class TrainRequest(StrictModel):
    base_model: str | None = None
    dataset_ids: list[str] | None = None
    output_name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    dry_run: bool = False
    gpu_devices: list[str] | Literal["all"] = "all"
    multi_gpu: bool = True
    method: Literal["lora", "qlora"] = "qlora"
    epochs: float = Field(default=1, gt=0, le=5)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.001)
    batch_size: int = Field(default=1, ge=1, le=16)
    gradient_accumulation: int = Field(default=4, ge=1, le=128)
    max_seq_length: int = Field(default=2048, ge=64, le=32768)
    rank: int = Field(default=16, ge=1, le=128)
    expert_name: str | None = Field(default=None, min_length=1, max_length=120)
    expert_description: str = Field(default="", max_length=2000)
    expert_topics: list[str] = Field(default_factory=list, max_length=20)
    source_expert_ids: list[str] = Field(default_factory=list, max_length=30)
    expert_auto: bool = False


class AdaptiveSchedule(StrictModel):
    mode: Literal["until_admitted"] = "until_admitted"
    refresh_optimizer_steps: Literal[50] = 50
    initial_update_budget: Literal[50] = 50
    seed: int | None = Field(default=None, strict=True)


class SleepRequest(TrainRequest):
    evaluation_protocol: Literal["behavioral_v1"] = "behavioral_v1"
    dynamic: bool = True
    context_examples: bool = True
    min_verified_examples: int = Field(default=3, ge=1, le=5000)
    max_generated_examples: int = Field(default=32, ge=0, le=256)
    generation_model: str | None = Field(default=None, min_length=1, max_length=200)
    generation_thinking: ThinkingMode = "off"
    schedule: AdaptiveSchedule | None = None
    max_episodes: int = Field(default=200, ge=1, le=5000)
    scope: Literal["all", "last_7_days", "last_24_hours"] = "all"

    @model_validator(mode="after")
    def adaptive_teacher(self):
        if self.schedule is not None:
            if self.generation_thinking == "off":
                self.generation_thinking = "max"
            elif self.generation_thinking != "max":
                raise ValueError("Adaptive Sleep requires maximum teacher thinking")
        return self


class SleepResumeRequest(StrictModel):
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"\S")


class MeditationRequest(StrictModel):
    evaluation_protocol: Literal["behavioral_v1"] = "behavioral_v1"
    pilot_updates: int | None = Field(default=None, strict=True, ge=1, le=100)
    mode: Literal["supervised", "unsupervised"] = "supervised"
    base_model: str | None = Field(default=None, min_length=1, max_length=80)
    conversation_ids: list[str] | None = Field(default=None, max_length=5000)
    source_roles: dict[str, Literal["train", "development", "heldout"]] | None = Field(default=None, max_length=5000)
    seed: int = Field(default=19, strict=True)
    dry_run: bool = False
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"\S")
    max_seq_length: int = Field(default=512, ge=64, le=4096)

    @model_validator(mode="after")
    def selected_sources(self):
        if self.conversation_ids is not None and (not self.conversation_ids
                or any(not ident.strip() for ident in self.conversation_ids)
                or len(set(self.conversation_ids)) != len(self.conversation_ids)):
            raise ValueError("Choose unique conversation IDs or all eligible chats")
        return self


class MeditationMutation(StrictModel):
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"\S")


class MeditationResume(MeditationMutation):
    conversation_ids: list[str] | None = Field(default=None, max_length=5000)
    source_roles: dict[str, Literal["train", "development", "heldout"]] | None = Field(default=None, max_length=5000)

    @model_validator(mode="after")
    def selected_sources(self):
        if self.conversation_ids is not None and (not self.conversation_ids
                or any(not ident.strip() for ident in self.conversation_ids)
                or len(set(self.conversation_ids)) != len(self.conversation_ids)):
            raise ValueError("Choose unique conversation IDs when extending a stopped run")
        return self


class MeditationRollback(MeditationMutation):
    base_id: str | None = Field(default=None, min_length=1, max_length=128)


class ExampleReview(StrictModel):
    decision: Literal["approve", "reject"]
    content_hash: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    correction: str | None = Field(default=None, min_length=1, max_length=100000)
    reason: str | None = Field(default=None, max_length=2000)


class CancelRequest(StrictModel):
    run_id: str | None = None


class AgentRequest(StrictModel):
    goal: str = Field(min_length=1, max_length=100000)
    workspace: str | None = None
    access: Literal["read", "write", "execute", "computer"] = "read"
    dialect: Literal["chat_completions", "anthropic_messages", "responses"] = "chat_completions"
    model: str | None = None
    conversation_id: str | None = None
    thinking: ThinkingMode = "off"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=8192, ge=1)
    max_rounds: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    stop_when_done: bool = True
    continue_output: bool = True
    max_continuations: int | None = Field(default=None, ge=0)
    context_compaction: bool = True
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)
    require_approval: bool = False
    auto_approve_commands: bool = False
    use_memory: bool = False
    allow_images: bool = False
    images: list[str] = Field(default_factory=list, max_length=4)
    allow_mcp: bool = False
    mcp_servers: list[str] = Field(default_factory=list, max_length=20)
    response_format: dict | None = None

    @model_validator(mode="after")
    def valid_goal(self):
        if not self.goal.strip():
            raise ValueError("Provide an agent goal")
        if self.images:
            self.allow_images = True
        if self.mcp_servers:
            self.allow_mcp = True
        if self.response_format is not None:
            import json
            if len(json.dumps(self.response_format)) > 65536:
                raise ValueError("Response schema is too large")
            if self.response_format.get("type") not in {"json_object", "json_schema"}:
                raise ValueError("Agent response_format must be json_object or json_schema")
            from .providers import prepare_format
            try:
                prepare_format(self.response_format)
            except Exception as exc:
                raise ValueError("Invalid structured response format: " + str(exc)) from exc
        return self


class AgentApproval(StrictModel):
    approval_id: str = Field(min_length=1, max_length=100)
    approved: bool


class AgentInput(StrictModel):
    content: str = Field(min_length=1, max_length=100000)
    interrupt: bool = True
    options: dict = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class AgentResume(StrictModel):
    instruction: str | None = Field(default=None, min_length=1, max_length=100000)
    limits: dict = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class ExpertProposal(StrictModel):
    conversation_ids: list[str] | None = None
    base_model: str | None = None


class ExpertBuild(SleepRequest):
    # User-facing names are accepted without duplicating learning's internal contract.
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    topics: list[str] | None = Field(default=None, max_length=20)
    conversation_ids: list[str] | None = None

    def learning_request(self):
        data = self.model_dump(exclude_none=True)
        for alias, canonical in (("name", "expert_name"), ("description", "expert_description"),
                                  ("topics", "expert_topics"), ("conversation_ids", "dataset_ids")):
            if alias in data:
                data[canonical] = data.pop(alias)
        return data


class AgentMemoryRequest(StrictModel):
    summary: str | None = Field(default=None, min_length=1, max_length=16000)
