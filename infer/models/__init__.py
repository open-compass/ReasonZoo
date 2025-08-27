import importlib

class ModelLoader:
    def __init__(self, model_name, config, use_accel=False, code_mode='noncode'):
        self.model_name = model_name
        self.config = config
        self.use_accel = use_accel
        self.code_mode = code_mode
        self._model = None

    def _lazy_import(self, module_name, func_name):
        if module_name.startswith('.'):
            module_name = __package__ + module_name
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def load_model(self):
        if self._model is None:
            load_func = self._lazy_import(self.config['load'][0], self.config['load'][1])
            if 'api' in self.config.get('call_type'):
                self._model = load_func(
                    self.config['model_name'], 
                    self.config['base_url'], 
                    self.config['api_key'], 
                    self.config['model'],
                    self.config['call_type'],
                    self.code_mode
                )
            else:
                self._model = load_func(self.model_name, self.config, self.use_accel, self.code_mode)
        return self._model

    @property
    def model(self):
        return self.load_model()

    @property
    def infer(self):
        return self._lazy_import(self.config['infer'][0], self.config['infer'][1])

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, name, config, code_mode='noncode'):
        """Register a model configuration."""
        self.models[name] = ModelLoader(name, config, use_accel=False, code_mode=code_mode)

    def load_model(self, choice, use_accel=False, code_mode='noncode'):
        """Load a model based on the choice."""
        if choice in self.models:
            self.models[choice].use_accel = use_accel
            self.models[choice].code_mode = code_mode
            return self.models[choice].model
        else:
            raise ValueError(f"Model choice '{choice}' is not supported.")

    def infer(self, choice, code_mode='noncode'):
        """Get the inference function for a given model."""
        if choice in self.models:
            self.models[choice].code_mode = code_mode
            return self.models[choice].infer
        else:
            raise ValueError(f"Inference choice '{choice}' is not supported.")

# Initialize model registry
model_registry = ModelRegistry()

# Configuration of models
model_configs = {
    ####### APi models #######
    'gpt-4o': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_name': 'GPT4o',
        'base_url': '',
        'api_key': '',
        'model': 'gpt-4o-2024-05-13',
        'call_type': 'api_chat'
    },
    'Deepseek-R1': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_name': 'Deepseek-R1',
        'base_url': '',
        'api_key': '',
        'model': 'deepseek-r1',
        'call_type': 'api_chat'
    },
    
    ####### Local Language Aligned models #######
    'Qwen2.5-0.5B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_name': 'Qwen/Qwen2.5-0.5B-Instruct',
        'call_type': 'local',
        'tp': 1
    }
}

# # Register all models
# for model_name, config in model_configs.items():
#     model_registry.register_model(model_name, config)

def load_model(choice, use_accel=False, code_mode='noncode'):
    """Load a specific model based on the choice."""
    model_registry.register_model(choice, model_configs[choice], code_mode=code_mode)
    return model_registry.load_model(choice, use_accel, code_mode=code_mode)

def infer(choice):
    """Get the inference function for a specific model."""
    return model_registry.infer(choice)

