from ratsnlp.nlpbook.generation import get_web_service_app
from kogpt2_service import inference_fn

if __name__ == '__main__':
    app = get_web_service_app(inference_fn)
    app.run()
