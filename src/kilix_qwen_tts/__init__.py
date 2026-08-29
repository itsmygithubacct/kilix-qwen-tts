"""Engine-neutral candidate mechanics for kilix-qwen-tts."""

from .surface import (
    CLI_COMMANDS,
    PROVIDER_REFUSAL,
    RUNTIME_COMMANDS,
    AudioFormat,
    AudioStreamAssembler,
    ConsentBinding,
    JobLifecycle,
    JobState,
    PromptAudio,
    RuntimeUnselected,
    SurfaceError,
    SynthesisResult,
    bind_prompt,
    inspect_command,
    render_wav,
)

__all__ = [
    "CLI_COMMANDS",
    "PROVIDER_REFUSAL",
    "RUNTIME_COMMANDS",
    "AudioFormat",
    "AudioStreamAssembler",
    "ConsentBinding",
    "JobLifecycle",
    "JobState",
    "PromptAudio",
    "RuntimeUnselected",
    "SurfaceError",
    "SynthesisResult",
    "bind_prompt",
    "inspect_command",
    "render_wav",
]
