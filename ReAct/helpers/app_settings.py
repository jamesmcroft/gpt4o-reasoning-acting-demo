class AppSettings:
    def __init__(self, config: dict):
        self.openai_endpoint = config['OPENAI_ENDPOINT']
        self.gpt4o_model_deployment_name = config['GPT4O_MODEL_DEPLOYMENT_NAME']
        self.text_embedding_model_deployment_name = config['TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME']
