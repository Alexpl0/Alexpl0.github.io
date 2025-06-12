import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell
import PyPDF2
import streamlit as st
import tempfile
import os

def pdf_to_ipynb(pdf_path, ipynb_path):
    # Leer el PDF
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

    # Crear un notebook con una celda markdown
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(text))

    # Guardar el notebook
    with open(ipynb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

# Interfaz Streamlit
st.title("Convertidor PDF a Jupyter Notebook (.ipynb)")

uploaded_file = st.file_uploader("Sube tu archivo PDF", type=["pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tmp_pdf_path = tmp_pdf.name

    ipynb_path = tmp_pdf_path.replace(".pdf", ".ipynb")
    pdf_to_ipynb(tmp_pdf_path, ipynb_path)

    with open(ipynb_path, "rb") as f:
        st.download_button(
            label="Descargar archivo .ipynb",
            data=f,
            file_name="convertido.ipynb",
            mime="application/x-ipynb+json"
        )

    # Limpieza de archivos temporales
    os.remove(tmp_pdf_path)
    os.remove(ipynb_path)