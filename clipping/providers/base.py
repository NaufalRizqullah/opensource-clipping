"""Provider errors contain safe diagnostics, never prompts or response bodies."""


class ProviderError(ValueError):
    def __init__(self, message, *, status_code=None, retryable=False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ProviderConfigError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class ContextOverflowError(ProviderError):
    pass


class InvalidOutputError(ProviderError):
    pass
