import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.title("Форматирование курсовой работы")

uploaded = st.file_uploader("Загрузите .docx файл", type=["docx"])

if uploaded and st.button("Форматировать"):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.docx"
        output_path = Path(tmpdir) / "output.docx"
        input_path.write_bytes(uploaded.getvalue())

        try:
            with st.spinner("Загрузка модели и классификация абзацев…"):
                from format_docx import format_document
                format_document(str(input_path), str(output_path), "format_config.yaml")
            st.session_state["result_bytes"] = output_path.read_bytes()
            st.session_state["result_name"] = f"formatted_{uploaded.name}"
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.session_state.pop("result_bytes", None)

if "result_bytes" in st.session_state:
    st.success("Готово!")
    st.download_button(
        label="Скачать отформатированный файл",
        data=st.session_state["result_bytes"],
        file_name=st.session_state["result_name"],
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
