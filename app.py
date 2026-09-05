import os
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def iniciar_agente():
    print("Cargando la base de conocimientos de NutriFit...")
    loader_csv = CSVLoader(file_path="./data/inventario.csv", encoding="utf-8")
    documentos = loader_csv.load()
    
    print("Vectorizando datos localmente...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=documentos, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return retriever

if __name__ == "__main__":
    retriever = iniciar_agente()
    print("\n--- NutriBot (Modo Local Gratuito) Iniciado. Escribe 'salir' para terminar ---")
    while True:
        pregunta = input("\nEjecutivo NutriFit: ")
        if pregunta.lower() == "salir":
            break
        
        # Búsqueda directa en la base vectorial sin gastar créditos
        docs_encontrados = retriever.invoke(pregunta)
        
        print("\nNutriBot:")
        if docs_encontrados:
            print("Basado en el inventario de NutriFit Chile, encontré la siguiente información:")
            for doc in docs_encontrados:
                print(f"- {doc.page_content}")
        else:
            print("No poseo esa información en mis registros actuales.")

