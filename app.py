from ctransformers import AutoModelForCausalLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from llama_cpp import Llama

PATH_MODEL_BIN = ".\models\llama-2-7b-chat.ggmlv3.q4_0.bin"
PATH_MODEL_GGUF = ".\models\Phi-3-mini-4k-instruct-q4.gguf"

# Carregar o modelo do arquivo GGUF
def loadModel():
    llm = Llama(model_path=PATH_MODEL_GGUF, verbose=True)
    return llm


# Carregar o modelo do arquivo BIN
def loadModelWithCtransformers():
    llm = AutoModelForCausalLM.from_pretrained(
        PATH_MODEL_BIN,
        model_type='llama')
    return llm


# Gerar texto
def generateText():
    llm = Llama(
        model_path=PATH_MODEL_GGUF,
        n_ctx=16000,    # Context length do texto
        n_threads=8,    # Número de threads de CPU a serem usados
        n_gpu_layers=0  # Número de GPUs a serem usadas
    )

    generation_kwargs = {
        "max_tokens": 20000,
        "stop": ["</s>"],
        "echo": False,
        "top_k": 1 
    }

    prompt = "O que é um LLM?"
    res = llm(prompt, **generation_kwargs)

    print(res["choices"][0]["text"])


# Gerar texto com Ctransformers e LLMChain
def generateTextWithCtransformers():
    template = """Question: {question}
    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm = CTransformers(model=PATH_MODEL_BIN, model_type='llama')
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    response = llm_chain.run("O que é um LLM?")

    print(response)


# Gerar texto com Ctransformers e stream
def generateTextWithCtransformersStream(question):
    llm = loadModelWithCtransformers()
    for text in llm(f"{question}. Responda sempre em pt-BR e em uma frase curta.", stream=True):
        print(text, end="", flush=True)


# Carregar o modelo do arquivo GGUF e gerar texto com chat completion
def generateTextWithGGUFChat():
    llm = Llama(model_path=PATH_MODEL_GGUF, chat_format="chatml", verbose=False)
    resposta = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "Você é um assistente de perguntas e respostas.",
            },
            {"role": "user", "content": "O que são chatbots?"},
        ],
        temperature=0.3,
    )
    print(resposta)


# generateTextWithCtransformers()
# generateTextWithCtransformersStream("O que é um LLM?")
generateTextWithGGUFChat()