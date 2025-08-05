#bu llm ile algılayıp validayson yapması
import base64
import fitz
import json
import pandas as pd
from typing import TypedDict, Optional, List
import gradio as gr
from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
import datetime
import re
import traceback

# --- API Anahtarları ve Model İsimleri ---
OPENROUTER_API_KEY = "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"
DOCUMENT_EXTRACTION_AGENT = "openai/gpt-4o"
CONTEXTUAL_VALIDATOR_AGENT = "openai/gpt-4o"
THOUGHT_CHAIN_AGENT = "google/gemini-flash-1.5"
SCHEMA_GENERATOR_AGENT = "openai/gpt-4o"


# --- PROGRAMATİK DOĞRULAMA ARAÇLARI ---
def _validate_tckn(tckn: str) -> str:
    if not isinstance(tckn, str): return "⚠️ Geçersiz veri tipi."
    tckn = tckn.strip()
    if not tckn.isdigit() or len(tckn) != 11 or tckn.startswith('0'): return f"❌ Geçersiz TCKN: '{tckn}'"
    sum1, sum2 = sum(map(int, tckn[0:9:2])), sum(map(int, tckn[1:8:2]))
    if (sum1 * 7 - sum2) % 10 != int(tckn[9]): return f"❌ Algoritmik olarak geçersiz TCKN: '{tckn}'"
    if sum(map(int, tckn[0:10])) % 10 != int(tckn[10]): return f"❌ Algoritmik olarak geçersiz TCKN: '{tckn}'"
    return f"✅ TCKN geçerli: '{tckn}'"


def _validate_iban(iban: str) -> str:
    if not isinstance(iban, str): return "⚠️ Geçersiz veri tipi."
    iban_clean = iban.replace(" ", "").upper()
    if not iban_clean.startswith("TR") or len(iban_clean) != 26 or not iban_clean[
                                                                       2:].isdigit(): return f"❌ Geçersiz IBAN: '{iban}'"
    return f"✅ IBAN formatı geçerli: '{iban}'"


def _validate_vergi_no(vergi_no: str) -> str:
    if not isinstance(vergi_no, str): return "⚠️ Geçersiz veri tipi."
    if len(vergi_no.replace(" ", "")) == 10 and vergi_no.replace(" ",
                                                                 "").isdigit(): return f"✅ Vergi No geçerli: '{vergi_no}'"
    return f"❌ Geçersiz Vergi No: '{vergi_no}' (10 haneli rakam olmalı)"


def _validate_email(email: str) -> str:
    if not isinstance(email, str): return "⚠️ Geçersiz veri tipi."
    if re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email): return f"✅ E-posta geçerli: '{email}'"
    return f"❌ Geçersiz E-posta: '{email}'"


def _validate_phone(phone: str) -> str:
    if not isinstance(phone, str): return "⚠️ Geçersiz veri tipi."
    if re.match(r"^(?:\+90|0)?\s*5[0-9]{2}\s*[0-9]{3}\s*[0-9]{2}\s*[0-9]{2}$",
                phone.strip()): return f"✅ TR Telefon geçerli: '{phone}'"
    return f"❌ Geçersiz TR Telefon: '{phone}'"


def _validate_date(date_str: str) -> str:
    if not isinstance(date_str, str): return "⚠️ Geçersiz veri tipi."
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            datetime.datetime.strptime(date_str.strip(), fmt)
            return f"✅ Tarih geçerli: '{date_str}'"
        except ValueError:
            pass
    return f"❌ Geçersiz Tarih: '{date_str}'"


def _validate_numeric_amount(amount: any) -> str:
    if isinstance(amount, (int, float)): return f"✅ Sayısal format geçerli: {amount}"
    if not isinstance(amount, str): return "⚠️ Geçersiz veri tipi."
    try:
        float(amount.replace(",", "."))
        return f"✅ Sayısal format geçerli: '{amount}'"
    except ValueError:
        return f"❌ Geçersiz sayısal format: '{amount}'"


def _validate_postal_code(code: str) -> str:
    if not isinstance(code, str): return "⚠️ Geçersiz veri tipi."
    if len(code.strip()) == 5 and code.strip().isdigit(): return f"✅ Posta Kodu geçerli: '{code}'"
    return f"❌ Geçersiz Posta Kodu: '{code}' (5 haneli rakam olmalı)"


def _validate_gtip(code: str) -> str:
    if not isinstance(code, str): return "⚠️ Geçersiz veri tipi."
    if len(code.strip()) == 12 and code.strip().isdigit(): return f"✅ GTİP geçerli: '{code}'"
    return f"❌ Geçersiz GTİP: '{code}' (12 haneli rakam olmalı)"


VALIDATION_TOOLS_MAP = {
    "tckn_validator": _validate_tckn, "iban_validator": _validate_iban, "vergi_no_validator": _validate_vergi_no,
    "email_validator": _validate_email, "phone_validator": _validate_phone, "date_validator": _validate_date,
    "numeric_amount_validator": _validate_numeric_amount, "postal_code_validator": _validate_postal_code,
    "gtip_validator": _validate_gtip
}


class VLMAnalysisResult(BaseModel):
    extracted_data: str = Field(description="JSON formatında çıkarılan veri metni.")
    extraction_analysis: str = Field(description="Çıkarım süreci hakkında VLM'in analizi ve düşünceleri.")


def file_to_base64_images(filepath: str) -> List[str]:
    images_base64 = []

    def pil_image_to_base64(pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    try:
        if filepath.lower().endswith('.pdf'):
            with fitz.open(filepath) as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=200);
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images_base64.append(pil_image_to_base64(img))
        else:
            img = Image.open(filepath);
            images_base64.append(pil_image_to_base64(img))
    except Exception as e:
        print(f"Dosya okuma hatası: {e}");
        return []
    return images_base64


class GraphState(TypedDict):
    filepath: str;
    user_defined_schema: str;
    extraction_analysis: Optional[str];
    extracted_data: Optional[dict];
    validation_results: Optional[dict];
    agent_outcome: any;
    formatted_output: any;
    error_message: Optional[str];
    base64_images: Optional[List[str]]


def initial_data_prep_node(state: GraphState) -> GraphState:
    try:
        base64_images = file_to_base64_images(state["filepath"])
        if not base64_images: return {"error_message": "Dosya okunamadı veya desteklenmeyen formatta."}
        return {"base64_images": base64_images}
    except Exception as e:
        return {"error_message": f"Dosya hazırlama hatası: {e}"}


def extract_with_vlm_node(state: GraphState) -> GraphState:
    try:
        llm = ChatOpenAI(model_name=DOCUMENT_EXTRACTION_AGENT, openai_api_key=OPENROUTER_API_KEY,
                         openai_api_base="https://openrouter.ai/api/v1", temperature=0.0, max_tokens=4096)
        structured_llm = llm.with_structured_output(VLMAnalysisResult)
        system_prompt = "You are a world-class AI assistant specialized in document analysis and data extraction. Your task is to analyze the provided document image(s) and extract **ONLY the information requested in the user-defined JSON schema**. Do not guess or add any information that is not explicitly asked for in the schema. If a field from the schema is not present in the document, you must omit it or set its value to null. Format your output strictly according to the provided schema. Also, provide a brief analysis of your extraction process."
        human_prompt_template = [
            {"type": "text",
             "text": "Here is the user-defined JSON schema you MUST follow:\n```json\n{user_defined_schema}\n```"},
            *({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in
              state["base64_images"])
        ]
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt_template)])
        chain = prompt | structured_llm
        result: VLMAnalysisResult = chain.invoke({"user_defined_schema": state['user_defined_schema']})
        parsed_data = json.loads(result.extracted_data)
        data_to_return = parsed_data[0] if isinstance(parsed_data, list) and parsed_data else parsed_data
        if not isinstance(data_to_return, dict):
            return {"error_message": f"VLM'den beklenen format (dictionary) alınamadı."}
        return {"extracted_data": data_to_return, "extraction_analysis": result.extraction_analysis}
    except Exception as e:
        return {"error_message": f"VLM Düğümünde Hata: {e}\n{traceback.format_exc()}"}


def hybrid_validator_node(state: GraphState) -> dict:
    if state.get('error_message') or not state.get('extracted_data'): return {}
    schema = json.loads(state['user_defined_schema']);
    data = state['extracted_data']
    base64_images = state['base64_images'];
    all_validation_notes = {}
    validator_llm = ChatOpenAI(model_name=CONTEXTUAL_VALIDATOR_AGENT, openai_api_key=OPENROUTER_API_KEY,
                               openai_api_base="https://openrouter.ai/api/v1", temperature=0.0)
    validation_prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a meticulous validation assistant. Your job is to determine if a specific piece of data, extracted from a document, meets a given validation criterion. You must base your decision ONLY on the provided document context. Respond with '✅ GEÇERLİ' or '❌ GEÇERSİZ' followed by a brief, one-sentence explanation."),
        ("human", [
            {"type": "text", "text": "Please analyze the document provided as an image."},
            {"type": "text", "text": "Extracted Value: `{extracted_value}`"},
            {"type": "text", "text": "Validation Rule: `{validation_rule}`"},
            {"type": "text",
             "text": "Question: Is this extracted value correct according to the validation rule and the document's content?"},
            *([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in base64_images])
        ])
    ])
    validation_chain = validation_prompt_template | validator_llm

    def find_and_validate(schema_level, data_level, path=""):
        if not isinstance(schema_level, dict) or not isinstance(data_level, dict): return
        for key, schema_value in schema_level.items():
            if key not in data_level or data_level[key] is None: continue
            current_path = f"{path}.{key}" if path else key;
            data_value = data_level[key]
            if isinstance(schema_value, dict) and "_validation" in schema_value:
                validator_rule = schema_value["_validation"]
                if validator_rule in VALIDATION_TOOLS_MAP:
                    result = VALIDATION_TOOLS_MAP[validator_rule](data_value)
                    all_validation_notes[current_path] = f"[Programatik] {result}"
                else:
                    try:
                        llm_response = validation_chain.invoke(
                            {"extracted_value": str(data_value), "validation_rule": validator_rule})
                        all_validation_notes[current_path] = f"[Anlamsal] {llm_response.content}"
                    except Exception as e:
                        all_validation_notes[
                            current_path] = f"[Anlamsal] ❌ HATA: Doğrulama yapılırken sorun oluştu: {e}"
            elif isinstance(schema_value, dict) and isinstance(data_value, dict):
                find_and_validate(schema_value, data_value, current_path)
            elif isinstance(schema_value, list) and isinstance(data_level.get(key), list):
                if schema_value and isinstance(schema_value[0], dict):
                    for i, item in enumerate(data_level[key]):
                        if isinstance(item, dict): find_and_validate(schema_value[0], item, f"{current_path}[{i}]")

    find_and_validate(schema, data)
    return {"validation_results": all_validation_notes}


def final_report_node(state: GraphState) -> dict:
    if state.get("error_message"): return {}
    final_report = "## 📝 Akıllı Belge Analiz Raporu\n\n"
    if state.get("extraction_analysis"):
        final_report += f"### 💬 VLM'in İlk Analizi\n*'{state['extraction_analysis']}'*\n\n"
    if state.get("validation_results"):
        final_report += "### ⚙️ Doğrulama Sonuçları\n"
        notes = "\n".join([f"- `{k}`: {v}" for k, v in state['validation_results'].items()])
        final_report += f"{notes}\n\n"
    else:
        final_report += "### ⚙️ Doğrulama Sonuçları\nHerhangi bir doğrulama kuralı çalıştırılmadı.\n\n"
    final_report += "### ✅ Nihai Çıktı (JSON)\n"
    clean_data = state.get('extracted_data', {}).copy()
    final_report += f"```json\n{json.dumps(clean_data, indent=2, ensure_ascii=False)}\n```"
    return {"formatted_output": final_report}


workflow = StateGraph(GraphState)
workflow.add_node("initial_data_prep", initial_data_prep_node)
workflow.add_node("extract_with_vlm", extract_with_vlm_node)
workflow.add_node("hybrid_validator", hybrid_validator_node)
workflow.add_node("final_report", final_report_node)
workflow.set_entry_point("initial_data_prep")


def should_continue(state: GraphState) -> str:
    return "extract_with_vlm" if not state.get("error_message") else END


workflow.add_conditional_edges("initial_data_prep", should_continue)
workflow.add_edge("extract_with_vlm", "hybrid_validator")
workflow.add_edge("hybrid_validator", "final_report")
workflow.add_edge("final_report", END)
try:
    app = workflow.compile()
except GraphRecursionError:
    app = workflow.compile(checkpointer=None, recursion_limit=50)


# --- DÜZELTME: GELİŞMİŞ ŞABLON OLUŞTURUCU PROMPT'U ---
def generate_schema_from_chat(message: str, history: list):
    history.append([message, None])
    yield history, ""

    try:
        llm = ChatOpenAI(model_name=SCHEMA_GENERATOR_AGENT, openai_api_key=OPENROUTER_API_KEY,
                         openai_api_base="https://openrouter.ai/api/v1", temperature=0.0)
        available_validators_str = ", ".join(VALIDATION_TOOLS_MAP.keys())

        system_prompt = (
            "Sen, kullanıcıdan gelen serbest metin sorgusunu, bir belgedeki en alakalı alanlarla eşleştiren ve bunları bir JSON listesi olarak sunan, gelişmiş bir 'Anlamsal Arama ve Sıralama Motoru'sun. Senin görevin, basit anahtar kelime eşleştirmesinin ötesine geçerek, sorgunun anlamını ve niyetini anlamaktır."

            #"Sen, doğal dilde ifade edilmiş veri doğrulama kurallarını analiz edip, bunları belirli bir JSON formatına çeviren uzman bir yapay zekasın. Temel görevin, bir isteğin yapısını anlayarak iki ana doğrulama türü arasında ayrım yapmak ve buna uygun JSON çıktısını oluşturmaktır."
            "1. Programatik (Format/Yapısal) Doğrulama:"
            "Eğer kullanıcının isteği, bir alanın evrensel olarak bilinen bir formata veya veri tipine uygunluğunu denetlemeye yönelikse, _validation değeri olarak sana sunulan önceden tanımlanmış anahtar kelimelerden uygun olanını seçmelisin. Bu tür doğrulamalar, alanın kendi değeri dışında başka bir bilgiye ihtiyaç duymaz."
            f"- Kullanabileceğin anahtar kelimeler şunlardır: {available_validators_str}\n"
            "- Uygulanacak Şema:"
            "json" "{" ' "<alan_adi_1>": {' ' "_value": "<beklenen_veri_tipi>",' ' "_validation": "<listeden_seçilen_validator_adi>"' ' },' ' "<alan_adi_2>": {' ' "_value": "<beklenen_veri_tipi>",' ' "_validation": "<listeden_seçilen_başka_bir_validator>"' ' }' "}" ""
            "2. Anlamsal (Bağlamsal/Mantıksal) Doğrulama:"
            "Eğer kullanıcının isteği, bir alanın değerinin belgedeki başka bir bilgiyle olan mantıksal ilişkisine veya belirli bir iş kuralına dayanıyorsa, _validation değeri olarak bu kuralı, başka bir yapay zekanın sorgusuz sualsiz anlayabileceği şekilde, net, tam ve kendi kendine yeterli bir cümle olarak formüle etmelisin. Bu tür doğrulamalar, birden fazla alan arasındaki ilişkiye veya genel bağlama bağımlıdır."
            "- Uygulanacak Şema:"
            "json" "{" ' "<alan_adi_A>": {' ' "_value": "<beklenen_veri_tipi>",' ' "_validation": "<alan_A\'nın uyması gereken mantıksal kuralın net ve tam açıklaması>"' ' },' ' "<alan_adi_B>": {' ' "_value": "<beklenen_veri_tipi>",' ' "_validation": "<alan_B\'nin, belgedeki başka bir alanla olan ilişkisini tanımlayan kural>"' ' }' "}" ""
            "Kural: Yanıtın yalnızca ve yalnızca istenen JSON nesnesini içermelidir. JSON bloğunun dışına asla bir açıklama, selamlama veya herhangi bir metin ekleme."
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]
        response = llm.invoke(messages)
        clean_response = response.content.strip().lstrip('```json').rstrip('```').strip()
        history[-1][1] = f"```json\n{clean_response}\n```"
    except Exception as e:
        history[-1][1] = f"Şablon oluşturulurken hata oluştu: {e}"

    yield history, ""


def run_agentic_workflow(document_file, schema_input, progress=gr.Progress()):
    if not document_file:
        yield [{"role": "assistant", "content": "❌ **Hata:**\nLütfen bir belge yükleyin."}];
        return
    try:
        json.loads(schema_input)
    except json.JSONDecodeError:
        yield [{"role": "assistant", "content": "❌ **Hata:**\nGeçerli bir JSON şablonu girilmedi."}];
        return
    progress(0, desc="Başlatılıyor...")
    chatbot_history = []
    initial_state = {"filepath": document_file.name, "user_defined_schema": schema_input}
    final_report_message = None
    try:
        for chunk in app.stream(initial_state, {"recursion_limit": 50}):
            node_name, node_output = list(chunk.items())[0]
            if not node_output: continue
            if node_output.get("error_message"):
                chatbot_history.append({"role": "assistant",
                                        "content": f"❌ **Hata Oluştu ({node_name}):**\n\n`{node_output['error_message']}`"})
                yield chatbot_history
                return
            if node_name == "final_report":
                final_report_message = node_output.get("formatted_output")
        if final_report_message:
            chatbot_history.append({"role": "assistant", "content": final_report_message})
        else:
            if not any("Hata" in msg["content"] for msg in chatbot_history):
                chatbot_history.append(
                    {"role": "assistant", "content": "Akış tamamlandı ancak nihai rapor oluşturulamadı."})
    except Exception as e:
        chatbot_history.append(
            {"role": "assistant", "content": f"❌ **Beklenmedik Sistem Hatası:**\n\n`{e}\n{traceback.format_exc()}`"})
    yield chatbot_history


if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Soft(), title="Akıllı Belge Sorgulama Agent'ı") as iface:
        gr.Markdown("# 🧠 Akıllı Belge Sorgulama Agent'ı")
        gr.Markdown(
            "Doğal dilde konuşarak JSON şablonu oluşturun veya anlamsal doğrulama kuralları ekleyin. Agent, sizin için bilgileri bulup hem formata hem de bağlama göre doğrulayacaktır.")
        with gr.Accordion("Adım 1: Şablon Oluşturma Yardımcısı", open=False):
            gr.Markdown(
                "İsteğinizi yazın, sizin için bir şablon oluşturalım.\n**Format Kuralı Örneği:** `TCKN ve e-posta istiyorum, ikisini de doğrula`\n `fatura toplamını istiyorum ve bu toplamın ara toplam ile kdv toplamına eşit olup olmadığını doğrula`")
            schema_chatbot = gr.Chatbot(label="Şablon Oluşturucu")
            schema_msg = gr.Textbox(label="İsteğinizi Buraya Yazın", placeholder="İsteğinizi yazıp Enter'a basın...")
            schema_msg.submit(generate_schema_from_chat, [schema_msg, schema_chatbot], [schema_chatbot, schema_msg])
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Adım 2: Belge Yükle", file_types=['.pdf', '.png', '.jpg', '.jpeg'])
                fields_input = gr.Code(
                    label="Adım 3: JSON Şablonunu Girin ",
                    value='{\n  "alici_adi": {\n    "_value": "string",\n    "_validation": "Bu alan belgedeki \'Yetkili Kişi\' altında yazan isimle aynı olmalıdır."\n  },\n  "staj_baslangic_tarihi": {\n    "_value": "string",\n    "_validation": "date_validator"\n  }\n}',
                    language="json",
                    lines=15
                )
                submit_btn = gr.Button("Analiz Et ve Raporla", variant="primary")
            with gr.Column(scale=2):
                output_chatbot = gr.Chatbot(label="Agent'ın Analiz Raporu", type="messages", height=700)
        submit_btn.click(fn=run_agentic_workflow, inputs=[file_input, fields_input], outputs=[output_chatbot],
                         concurrency_limit=1)

    iface.launch(share=True, debug=True)
