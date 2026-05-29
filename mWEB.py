import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import unicodedata
from scipy.optimize import minimize_scalar
from processamento import balanceProcessing, jumpProcessing, tugProcessing, ytestProcessing, jointSenseProcessing
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.integrate import trapezoid
from scipy.ndimage import uniform_filter1d
from textwrap import dedent
from scipy.signal import butter, filtfilt, detrend
import io

# --------- Config da página ---------
st.set_page_config(page_title="Momentum Web", page_icon="⚡", layout="wide")

# --------- Estilo ---------
st.markdown(
    """
    <style>
      .stApp {
        background-color: var(--background-color);
      }
      header[data-testid="stHeader"] {
        background-color: var(--background-color) !important;
      }
      .block-container { background: transparent; }
      section[data-testid="stSidebar"] { background-color: var(--secondary-background-color); }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Momentum Web</h1>", unsafe_allow_html=True)

# =========================
#  CORREÇÃO DO CARREGAMENTO
# =========================

AUDIO_EXTS = {"3ga", "aac", "m4a", "mp3", "wav", "ogg", "flac"}
MAX_UNIT_SUFFIX_LENGTH = 3
SIX_COL_ELAPSED_INDEX = 2
SIX_COL_X_INDEX = 3
SIX_COL_Y_INDEX = 4
SIX_COL_Z_INDEX = 5
ELAPSED_COLUMN_NAMES = ["elapsed", "tempo"]
EPOCH_COLUMN_NAMES = ["epoch"]
X_AXIS_COLUMN_NAMES = ["xaxis", "accx", "aceleraçãox", "xg"]
Y_AXIS_COLUMN_NAMES = ["yaxis", "accy", "aceleraçãoy", "yg"]
Z_AXIS_COLUMN_NAMES = ["zaxis", "accz", "aceleraçãoz", "zg"]

def _parece_audio_3gp_mp4(head: bytes) -> bool:
    return len(head) >= 12 and head[4:8] == b"ftyp"

def carregar_dados_generico(arquivo):
    """
    - Detecta áudio por extensão ou assinatura e NÃO passa no pandas.
    - Para CSV: tenta encodings comuns (utf-8-sig, utf-16, cp1252, latin1).
    """
    try:
        nome = getattr(arquivo, "name", "arquivo").lower()
        ext = nome.rsplit(".", 1)[-1] if "." in nome else ""

        # lê bytes 1x
        try:
            arquivo.seek(0)
        except Exception:
            pass
        raw = arquivo.read()
        head = raw[:32]

        # Se for áudio (ou parecer 3gp/mp4), encaminhe para seu loader de áudio
        if (ext in AUDIO_EXTS) or _parece_audio_3gp_mp4(head):
            return carregar_dados_generico_audio_bytes(raw, nome, force_mono=True)  # <- sua função

        # tenta encodings
        if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
            encodings = ["utf-16"]
        else:
            encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

        last_err = None
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=None,
                    engine="python",
                    encoding=enc,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                df = None

        if df is None:
            raise last_err

        def _normalize_column_name(col):
            normalized = unicodedata.normalize("NFKD", str(col).strip().lower())
            normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == "_")
            return normalized.replace("epoc", "epoch")

        def _matches_target(column_name, target):
            if column_name == target:
                return True
            if column_name.startswith(target):
                suffix = column_name[len(target):]
                return len(suffix) <= MAX_UNIT_SUFFIX_LENGTH
            return False

        def _find_column(targets):
            normalized_targets = [_normalize_column_name(target) for target in targets]
            normalized_targets_compact = [
                target.replace("_", "") for target in normalized_targets
            ]
            for col in df.columns:
                nome_col = _normalize_column_name(col)
                nome_col_compact = nome_col.replace("_", "")
                for alvo, alvo_compact in zip(
                    normalized_targets, normalized_targets_compact
                ):
                    if _matches_target(nome_col, alvo) or _matches_target(
                        nome_col_compact, alvo_compact
                    ):
                        return col
            return None

        if df.shape[1] == 6:
            tempo_col = _find_column(ELAPSED_COLUMN_NAMES)
            if tempo_col is not None:
                tempo_ms = pd.to_numeric(df[tempo_col], errors="coerce") * 1000.0
            else:
                epoch_col = _find_column(EPOCH_COLUMN_NAMES)
                if epoch_col is not None:
                    tempo_ms = pd.to_numeric(df[epoch_col], errors="coerce")
                else:
                    # Fallback: assume 6-col layout [epoch, timestamp, elapsed (s), x, y, z].
                    # Expected unit: elapsed time in seconds, converting to ms here.
                    # Se o formato não estiver nesse padrão, revise os cabeçalhos do CSV.
                    tempo_ms = (
                        pd.to_numeric(df.iloc[:, SIX_COL_ELAPSED_INDEX], errors="coerce")
                        * 1000.0
                    )

            x_col = _find_column(X_AXIS_COLUMN_NAMES)
            y_col = _find_column(Y_AXIS_COLUMN_NAMES)
            z_col = _find_column(Z_AXIS_COLUMN_NAMES)
            # Fallback: assume x/y/z are the last three columns in the 6-col layout
            x_col = x_col or df.columns[SIX_COL_X_INDEX]
            y_col = y_col or df.columns[SIX_COL_Y_INDEX]
            z_col = z_col or df.columns[SIX_COL_Z_INDEX]

            dados = pd.DataFrame(
                {
                    "Tempo": tempo_ms,
                    "X": pd.to_numeric(df[x_col], errors="coerce"),
                    "Y": pd.to_numeric(df[y_col], errors="coerce"),
                    "Z": pd.to_numeric(df[z_col], errors="coerce"),
                }
            )
        elif df.shape[1] == 5:
            dados = df.iloc[:, 1:5].copy()
        elif df.shape[1] == 4:
            dados = df.iloc[:, 0:4].copy()
        else:
            st.error("O arquivo deve conter 4, 5 ou 6 colunas com cabeçalhos.")
            return None

        dados.columns = ["Tempo", "X", "Y", "Z"]
        return dados

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

def _formatar_resultado_txt(variaveis):
    resultado_txt = "Variável\tValor\n"
    for nome, valor in variaveis:
        resultado_txt += f"{nome}\t{valor}\n"
    return resultado_txt

def _calcular_variaveis_equilibrio(dados, startRec, endRec, fc):
    rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction = \
        balanceProcessing.processar_equilibrio(dados, startRec, endRec, 1, 1, fc)

    tempo_sel, ml_sel, ap_sel, freqs_sel, psd_ml_sel, psd_ap_sel = \
        balanceProcessing.processar_equilibrio(dados, startRec, endRec, 1, 0, fc)
    
    LF_power_ml = trapezoid(psd_ml_sel[0:4], freqs_sel[0:4])
    LF_power_ap = trapezoid(psd_ap_sel[0:4], freqs_sel[0:4])
    
    MF_power_ml = trapezoid(psd_ml_sel[5:19], freqs_sel[5:19])
    MF_power_ap = trapezoid(psd_ap_sel[5:19], freqs_sel[5:19])

    HF_power_ml = trapezoid(psd_ml_sel[20:79], freqs_sel[20:79])
    HF_power_ap = trapezoid(psd_ap_sel[20:79], freqs_sel[20:79])
    
    total_power_ml = trapezoid(psd_ml_sel, freqs_sel)
    total_power_ap = trapezoid(psd_ap_sel, freqs_sel)

    for index, val in enumerate(psd_ml_sel):
        if np.sum(psd_ml_sel[0:index]) > total_power_ml/2:
            freq_median_ml = freqs_sel[index-1]
            break

    for index, val in enumerate(psd_ap_sel):
        if np.sum(psd_ap_sel[0:index]) > total_power_ap/2:
            freq_median_ap = freqs_sel[index-1]
            break

    variaveis = [
        ("RMS ML", round(rms_ml, 6)),
        ("RMS AP", round(rms_ap, 6)),
        ("Desvio total", round(total_deviation, 6)),
        ("Área elipse", round(ellipse_area, 6)),
        ("Potência total ML", round(total_power_ml, 10)),
        ("Potência total AP", round(total_power_ap, 10)),
        ("Potência Baixa Freq ML", round(LF_power_ml, 10)),
        ("Potência Baixa Freq AP", round(LF_power_ap, 10)),
        ("Potência Média Freq ML", round(MF_power_ml, 10)),
        ("Potência Média Freq AP", round(MF_power_ap, 10)),
        ("Potência Alta Freq ML", round(HF_power_ml, 10)),
        ("Potência Alta Freq AP", round(HF_power_ml, 10)),
        ("Frequência mediana ML", round(freq_median_ml, 10)),
        ("Frequência mediana AP", round(freq_median_ap, 10)),
    ]
    return variaveis

def _montar_df_medias_equilibrio(resultados_individuais):
    nomes = []
    for variaveis in resultados_individuais:
        for nome, _ in variaveis:
            if nome not in nomes:
                nomes.append(nome)

    valores_por_execucao = []
    for variaveis in resultados_individuais:
        valores_por_execucao.append({nome: valor for nome, valor in variaveis})

    dados_df = {"variavel": nomes}
    for indice, valores in enumerate(valores_por_execucao, start=1):
        dados_df[f"valor_{indice}"] = [valores.get(nome, np.nan) for nome in nomes]

    df = pd.DataFrame(dados_df)
    colunas_valores = [col for col in df.columns if col.startswith("valor_")]
    df["media_3"] = df[colunas_valores].mean(axis=1, skipna=True)
    return df

# =========================
#      NAVEGAÇÃO / PÁGINAS
# =========================

pagina = st.sidebar.radio(
    "📂 Navegue pelas páginas",
    ["🏠 Página Inicial", "⬆️ Importar Dados", "📈 Visualização Gráfica", "📤 Exportar Resultados", "📖 Referências bibliográficas"]
)

# === Página Inicial ===
if pagina == "🏠 Página Inicial":
    html = dedent("""
    <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333;
                max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6);
                padding: 20px; border-radius: 8px;">
      <p><b>Bem-vindo ao Momentum Web</b>, a aplicação Web para análise de dados de protocolos de avaliação do <i>Momentum Sensors</i>.</p>
      <p>Os protocolos de análise dos dados são baseados em métodos usados em artigos científicos do grupo idealizador do Projeto Momentum.</p>
      <p>Alguns protocolos estarão em desenvolvimento e serão indicados quando for o caso.</p>
      <p>Utilize o <b>menu lateral</b> para navegar entre as diferentes etapas da análise.</p>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

# === Página de Importação ===
elif pagina == "⬆️ Importar Dados":
    st.title("⬆️ Importar Dados")
    col1, col2, col3 = st.columns([1, 0.2, 1])

    with col1:
        tipo_teste = st.selectbox(
            "Qual teste você deseja analisar?",
            ["Selecione...", "Registro inercial livre", "Registro de áudio livre", "Equilíbrio", "Salto", "TUG", "Propriocepção", "Y test"]
        )

        if tipo_teste != "Selecione...":
            st.session_state["tipo_teste"] = tipo_teste

            # --------- Inercial genérico ---------
            if tipo_teste in {"Registro inercial livre", "Equilíbrio", "Salto", "Propriocepção"}:
                st.subheader(f"📥 Importar dados: {tipo_teste}")
                if tipo_teste == "Equilíbrio":
                    modo_equilibrio = st.radio(
                        "Modo de processamento",
                        ["Três arquivos", "Individual"],
                        index=0,
                        key="equilibrio_modo_radio",
                    )
                    st.session_state["equilibrio_modo_processamento"] = (
                        "3_arquivos" if modo_equilibrio == "Três arquivos" else "individual"
                    )

                    if modo_equilibrio == "Individual":
                        arquivo = st.file_uploader(
                            "Selecione o arquivo (CSV)",
                            type=["csv"],
                            key="equilibrio_arquivo_individual",
                        )
                        if arquivo is not None:
                            dados = carregar_dados_generico(arquivo)
                            # aqui, "dados" será DataFrame (texto) ou dict (áudio) — mas nesse fluxo só aceitamos CSV
                            if isinstance(dados, dict):
                                st.error("Você enviou um arquivo de áudio nesse uploader. Use 'Registro de áudio livre'.")
                            elif dados is not None:
                                st.success("Dados carregados com sucesso")
                                st.session_state["dados"] = dados
                                st.session_state["tipo_teste"] = tipo_teste
                    else:
                        arquivos = st.file_uploader(
                            "Selecione os arquivos (CSV)",
                            type=["csv"],
                            accept_multiple_files=True,
                            key="equilibrio_arquivos_tres",
                        )
                        if arquivos:
                            if len(arquivos) != 3:
                                st.error("Selecione exatamente 3 arquivos CSV.")
                            else:
                                dados_lista = []
                                erro = False
                                for arquivo in arquivos:
                                    dados = carregar_dados_generico(arquivo)
                                    if isinstance(dados, dict):
                                        st.error("Você enviou um arquivo de áudio nesse uploader. Use 'Registro de áudio livre'.")
                                        erro = True
                                        break
                                    elif dados is None:
                                        st.error("Não foi possível carregar um dos arquivos CSV.")
                                        erro = True
                                        break
                                    dados_lista.append(dados)
                                if not erro and len(dados_lista) == 3:
                                    st.success("Dados carregados com sucesso")
                                    st.session_state["dados"] = dados_lista[0]
                                    st.session_state["equilibrio_dados_3"] = dados_lista
                                    st.session_state["equilibrio_resultados_individuais"] = []
                                    st.session_state["equilibrio_resultado_medias_df"] = None
                                    st.session_state["tipo_teste"] = tipo_teste
                else:
                    arquivo = st.file_uploader("Selecione o arquivo (CSV)", type=["csv"])
                    if arquivo is not None:
                        dados = carregar_dados_generico(arquivo)
                        # aqui, "dados" será DataFrame (texto) ou dict (áudio) — mas nesse fluxo só aceitamos CSV
                        if isinstance(dados, dict):
                            st.error("Você enviou um arquivo de áudio nesse uploader. Use 'Registro de áudio livre'.")
                        elif dados is not None:
                            st.success("Dados carregados com sucesso")
                            st.session_state["dados"] = dados
                            st.session_state["tipo_teste"] = tipo_teste

            # --------- Áudio ---------
            elif tipo_teste == "Registro de áudio livre":
                st.subheader("🎙️ Importar arquivo de áudio livre")
                arquivo = st.file_uploader("Escolha um arquivo de áudio", type=list(AUDIO_EXTS))
                if arquivo is not None:
                    dados_audio = carregar_dados_generico(arquivo)
                    # aqui deve vir dict do áudio
                    if isinstance(dados_audio, dict):
                        st.success("Áudio carregado com sucesso")
                        st.session_state["dados"] = dados_audio
                        st.session_state["tipo_teste"] = tipo_teste
                        st.write(f"Taxa de amostragem: {dados_audio['sr']} Hz | Duração: {dados_audio['duration_s']:.2f} s")
                    else:
                        st.error("Não consegui ler como áudio. Verifique FFmpeg/pydub no servidor.")

            # --------- TUG (acc + gyro) ---------
            elif tipo_teste == "TUG":
                st.subheader("📱 Importar dados do TUG")
                arq_acc = st.file_uploader("Selecione o arquivo do acelerômetro (CSV)", type=["csv"], key="tug_acc")
                if arq_acc is not None:
                    dados_acc = carregar_dados_generico(arq_acc)
                    if isinstance(dados_acc, dict):
                        st.error("Você enviou um arquivo de áudio no acelerômetro. Precisa ser CSV.")
                    elif dados_acc is not None:
                        st.success("Acelerômetro carregado!")
                        st.dataframe(dados_acc.head())
                        st.session_state["dados_acc"] = dados_acc
                        st.session_state["dados"] = dados_acc

                        arq_gyro = st.file_uploader("Selecione o arquivo do giroscópio (CSV)", type=["csv"], key="tug_gyro")
                        if arq_gyro is not None:
                            dados_gyro = carregar_dados_generico(arq_gyro)
                            if isinstance(dados_gyro, dict):
                                st.error("Você enviou um arquivo de áudio no giroscópio. Precisa ser CSV.")
                            elif dados_gyro is not None:
                                st.success("Giroscópio carregado!")
                                st.dataframe(dados_gyro.head())
                                st.session_state["dados_gyro"] = dados_gyro
                                st.session_state["tipo_teste"] = tipo_teste

            # --------- Y test (coluna + joelho) ---------
            elif tipo_teste == "Y test":
                st.subheader("📱 Importar dados do Y test")
                arq_coluna = st.file_uploader("Selecione o arquivo da coluna (CSV)", type=["csv"], key="y_coluna")
                if arq_coluna is not None:
                    dados_coluna = carregar_dados_generico(arq_coluna)
                    if isinstance(dados_coluna, dict):
                        st.error("Você enviou um arquivo de áudio na coluna. Precisa ser CSV.")
                    elif dados_coluna is not None:
                        st.success("Coluna carregada!")
                        st.dataframe(dados_coluna.head())
                        st.session_state["dados_acc_coluna"] = dados_coluna
                        st.session_state["dados"] = dados_coluna

                        arq_joelho = st.file_uploader("Selecione o arquivo do joelho (CSV)", type=["csv"], key="y_joelho")
                        if arq_joelho is not None:
                            dados_joelho = carregar_dados_generico(arq_joelho)
                            if isinstance(dados_joelho, dict):
                                st.error("Você enviou um arquivo de áudio no joelho. Precisa ser CSV.")
                            elif dados_joelho is not None:
                                st.success("Joelho carregado!")
                                st.dataframe(dados_joelho.head())
                                st.session_state["dados_acc_joelho"] = dados_joelho
                                st.session_state["tipo_teste"] = tipo_teste

        else:
            st.info("Selecione um tipo de teste para continuar.")

    with col3:
        st.markdown(
            "<div style='background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;'>"
            "<b>Dica:</b> se você subir um <code>.3ga</code> no uploader de CSV por engano, agora o app detecta "
            "pelo cabeçalho (<code>ftyp</code>) e evita que o pandas tente ler como UTF-8.</div>",
            unsafe_allow_html=True
        )

# === Página de Visualização Gráfica ===
elif pagina == "📈 Visualização Gráfica":
    st.title("📈 Visualização Gráfica")

    if "dados" not in st.session_state or "tipo_teste" not in st.session_state:
        st.info("Dados ou tipo de teste não definidos. Vá até a aba 'Importar Dados'.")
    else:
        tipo_teste = st.session_state["tipo_teste"]
        st.subheader(f"📊 Visualização - {tipo_teste}")

        # -------- Registro inercial livre --------
        if tipo_teste == "Registro inercial livre":
            dados = st.session_state["dados"]
            t = (dados["Tempo"].astype(float) / 1000.0).values
            x = dados["X"].astype(float).values
            y = dados["Y"].astype(float).values
            z = dados["Z"].astype(float).values

            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                detrend_command = st.checkbox("Fazer detrend?", False)
                separados = st.checkbox("Gráficos separados?", False)
                filtrado = st.checkbox("Filtrar passa-baixa?", False)

                if detrend_command:
                    x_new = detrend(x)
                    y_new = detrend(y)
                    z_new = detrend(z)
                else:
                    x_new, y_new, z_new = x, y, z

                if filtrado:
                    dt = np.diff(t)
                    dt = dt[np.isfinite(dt) & (dt > 0)]
                    fs_mean = 1.0 / float(np.median(dt))
                    nyquist = fs_mean / 2.0

                    max_cutoff = max(0.1, float(nyquist * 0.99))
                    default_cutoff = min(50.0, max_cutoff)

                    cutoff = st.number_input(
                        "Frequência de corte (Hz)",
                        min_value=0.1,
                        max_value=float(max_cutoff),
                        value=float(default_cutoff),
                        step=0.1
                    )

                    cutoff = float(np.clip(cutoff, 0.1, nyquist * 0.99))
                    wn = cutoff / nyquist
                    b, a = butter(4, wn, btype="low", analog=False)
                    x_new = filtfilt(b, a, x_new)
                    y_new = filtfilt(b, a, y_new)
                    z_new = filtfilt(b, a, z_new)

            with col2:
                if separados:
                    for sig, label in [(x_new, "X"), (y_new, "Y"), (z_new, "Z")]:
                        fig, ax = plt.subplots(figsize=(10, 3.5))
                        ax.plot(t, sig, linewidth=0.8)
                        ax.set_xlabel("Tempo (s)")
                        ax.set_ylabel("Amplitude")
                        ax.set_title(label)
                        ax.grid(True)
                        st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(t, x_new, label="X", linewidth=0.8)
                    ax.plot(t, y_new, label="Y", linewidth=0.8)
                    ax.plot(t, z_new, label="Z", linewidth=0.8)
                    ax.set_xlabel("Tempo (s)")
                    ax.set_ylabel("Amplitude")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

        # -------- Registro de áudio livre --------
        elif tipo_teste == "Registro de áudio livre":
            dados_audio = st.session_state["dados"]
            if not isinstance(dados_audio, dict):
                st.error("O dado em sessão não está no formato de áudio (dict). Reimporte o áudio.")
            else:
                to_mono = st.checkbox("Manter mono (recomendado)", value=True)
                # OBS: o mono já foi aplicado no loader; esse checkbox fica só como UI aqui.
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(dados_audio["t"], dados_audio["x"], linewidth=0.8)
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Amplitude (normalizada)")
                ax.grid(True)
                st.pyplot(fig)

        # -------- Equilíbrio --------
        elif tipo_teste == "Equilíbrio":
            dados = st.session_state["dados"]
            tempo, ml, ap, freqs, psd_ml, psd_ap = balanceProcessing.processar_equilibrio(dados, 0, 0, 0, 0, 8)
            max_val = len(tempo)

            col1, col2, col3 = st.columns(3)
            with col1:
                startRec = st.number_input("Indique o início do registro", value=0, step=1, max_value=max_val)
            with col2:
                endRec = st.number_input("Indique o final do registro", value=max_val, step=1, max_value=max_val)
            with col3:
                fc = st.number_input("Indique o filtro passa-baixa", value=8.0, step=0.1, max_value=40.0)

            st.session_state["intervalo"] = (startRec, endRec, fc)
            showRec = st.checkbox("Mostrar o dado original", value=True)

            tempo, ml, ap, freqs, psd_ml, psd_ap = balanceProcessing.processar_equilibrio(dados, 0, 0, 0, 0, 49)
            tempo_sel, ml_sel, ap_sel, freqs_sel, psd_ml_sel, psd_ap_sel = balanceProcessing.processar_equilibrio(dados, startRec, endRec, 1, 0, fc)

            if startRec > endRec:
                st.error("Início não pode ser maior que o final.")
            else:
                fig = plt.figure(figsize=(8, 10))
                gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.8, wspace=0.6)

                rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction = \
                    balanceProcessing.processar_equilibrio(dados, startRec, endRec, 1, 1, fc)

                ellipse = Ellipse(xy=(avg_x, avg_y), width=width, height=height, angle=angle, alpha=0.3, zorder=10)

                ax1 = fig.add_subplot(gs[0:2, 0:2])
                if showRec:
                    ax1.plot(ml, ap, linewidth=0.5)
                ax1.plot(ml_sel[startRec:endRec], ap_sel[startRec:endRec], linewidth=0.8)
                ax1.add_patch(ellipse)
                ax1.set_xlabel(r"Aceleração ML (m/s$^2$)", fontsize=8)
                ax1.set_ylabel(r"Aceleração AP (m/s$^2$)", fontsize=8)
                ax1.grid(True)

                ax2 = fig.add_subplot(gs[0, 2:])
                if showRec:
                    ax2.plot(tempo, ml, linewidth=0.5)
                ax2.plot(tempo_sel[startRec:endRec], ml_sel[startRec:endRec], linewidth=0.8)
                ax2.set_xlabel("Tempo (s)", fontsize=8)
                ax2.set_ylabel(r"Aceleração ML (m/s$^2$)", fontsize=8)
                ax2.grid(True)

                ax3 = fig.add_subplot(gs[1, 2:])
                if showRec:
                    ax3.plot(tempo, ap, linewidth=0.5)
                ax3.plot(tempo_sel[startRec:endRec], ap_sel[startRec:endRec], linewidth=0.8)
                ax3.set_xlabel("Tempo (s)", fontsize=8)
                ax3.set_ylabel(r"Aceleração AP (m/s$^2$)", fontsize=8)
                ax3.grid(True)

                ax4 = fig.add_subplot(gs[2:4, 0:2])
                if showRec:
                    ax4.plot(freqs, psd_ml, linewidth=0.5)
                ax4.plot(freqs_sel, psd_ml_sel, linewidth=0.8)
                ax4.set_xlabel("Frequência (Hz)", fontsize=8)
                ax4.set_ylabel("PSD ML", fontsize=8)
                ax4.set_xlim([0,8])
                ax4.set_ylim([0,0.005])
                ax4.grid(True)

                ax5 = fig.add_subplot(gs[2:4, 2:])
                if showRec:
                    ax5.plot(freqs, psd_ap, linewidth=0.5)
                ax5.plot(freqs_sel, psd_ap_sel, linewidth=0.8)
                ax5.set_xlabel("Frequência (Hz)", fontsize=8)
                ax5.set_ylabel("PSD AP", fontsize=8)
                ax5.set_xlim([0,8])
                ax5.set_ylim([0,0.005])
                ax5.grid(True)

                st.pyplot(fig)

        elif tipo_teste == "Salto":
            dados = st.session_state["dados"]
            tempo, salto, startJump, endJump, altura, tempo_voo, m1, m2, veloc, desloc, istart, iend = jumpProcessing.processar_salto(dados, 8)
            fig, ax = plt.subplots()
            ax.plot(tempo[max(0, istart-100):min(len(tempo), iend+100)],
                    salto[max(0, istart-100):min(len(salto), iend+100)], linewidth=0.8)
            ax.axvline(startJump, linestyle="--", linewidth=0.8, label="Início Voo")
            ax.axvline(endJump, linestyle="--", linewidth=0.8, label="Fim Voo")
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Aceleração vertical (m/s²)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        elif tipo_teste == "TUG":
            dados_acc = st.session_state["dados_acc"]
            dados_gyro = st.session_state["dados_gyro"]
            baseline_onset = st.number_input('Indique o momento inicial da baseline do início do teste (s)', value=0.0)
            baseline_offset = st.number_input('Indique o momento inicial da baseline do final do teste (s)', value=(np.max(dados_acc["Tempo"])/1000)-2.5)
            st.session_state["baseline_onset"] = baseline_onset
            st.session_state["baseline_offset"] = baseline_offset
            col1, col2 = st.columns(2)
            t_novo_acc, v_acc, ml_acc, z_acc_filtrado, norma_acc_filtrado, t_novo_gyro, v_gyro, ml_gyro, z_gyro_filtrado, norma_gyro_filtrado,start_test,stop_test,idx,idx_ml,idx_acc_ap,idx_acc_v,duration = tugProcessing.processar_tug(dados_acc,dados_gyro,2,1.25,baseline_onset,baseline_offset)
            vertical_squared = np.sqrt(v_gyro**2)
            lat1 = idx[1][0]
            lat2 = idx[1][1]
            amp1 = vertical_squared[idx[0][0]]
            amp2 = vertical_squared[idx[0][1]]
            if lat1 > lat2:
                G1_lat = lat2
                G1_amp = amp2
                G2_lat = lat1
                G2_amp = amp1
            else: 
                G1_lat = lat1
                G1_amp = amp1
                G2_lat = lat2
                G2_amp = amp2
            ml_squared = np.sqrt(ml_gyro**2)
            lat1 = idx_ml[1][0]
            lat2 = idx_ml[1][1]
            amp1 = ml_squared[idx_ml[0][0]]
            amp2 = ml_squared[idx_ml[0][1]]
            if lat1 > lat2:
                G0_lat = lat2
                G0_amp = amp2
                G4_lat = lat1
                G4_amp = amp1
            else: 
                G0_lat = lat1
                G0_amp = amp1
                G4_lat = lat2
                G4_amp = amp2
            acc_ap_squared = np.sqrt(z_acc_filtrado**2)
            lat1 = idx_acc_ap[1][0]
            lat2 = idx_acc_ap[1][1]
            amp1 = acc_ap_squared[idx_acc_ap[0][0]]
            amp2 = acc_ap_squared[idx_acc_ap[0][1]]
            if lat1 > lat2: 
                A1_lat = lat2 
                A1_amp = amp2 
                A2_lat = lat1 
                A2_amp = amp1 
            else: 
                A1_lat = lat1
                A1_amp = amp1
                A2_lat = lat2
                A2_amp = amp2
            acc_v_squared = np.sqrt(v_acc**2)
            lat1 = idx_acc_v[1][0]
            lat2 = idx_acc_v[1][1]
            amp1 = acc_v_squared[idx_acc_v[0][0]]
            amp2 = acc_v_squared[idx_acc_v[0][1]]
            if lat1 > lat2: 
                A1v_lat = lat2 
                A1v_amp = amp2 
                A2v_lat = lat1 
                A2v_amp = amp1 
            else: 
                A1v_lat = lat1 
                A1v_amp = amp1 
                A2v_lat = lat2 
                A2v_amp = amp2 
            with col1: 
                fig1, ax1 = plt.subplots()
                ax1.plot(t_novo_acc, norma_acc_filtrado, linewidth=0.8, color='black')
                ax1.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8) 
                ax1.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8) 
                ax1.set_xlim(start_test-5,stop_test+5) 
                ax1.set_xlabel('Tempo (s)') 
                ax1.set_ylabel('Aceleração norma (m/s²)') 
                ax1.legend() 
                st.pyplot(fig1) 
                fig2, ax2 = plt.subplots() 
                ax2.plot(t_novo_acc, np.sqrt(ml_acc**2), linewidth=0.8, color='black') 
                ax2.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8) 
                ax2.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8) 
                ax2.set_xlim(start_test-5,stop_test+5) 
                ax2.set_xlabel('Tempo (s)') 
                ax2.set_ylabel('Aceleração ML (m/s²)') 
                ax2.legend() 
                st.pyplot(fig2) 
                fig3, ax3 = plt.subplots() 
                ax3.plot(t_novo_acc, np.sqrt(v_acc**2), linewidth=0.8, color='black') 
                ax3.plot(A1v_lat,A1v_amp,'ro') 
                ax3.plot(A2v_lat,A2v_amp,'ro') 
                ax3.set_xlim(start_test-5,stop_test+5) 
                ax3.set_xlabel('Tempo (s)') 
                ax3.set_ylabel('Aceleração vertical (m/s²)') 
                ax3.legend() 
                st.pyplot(fig3) 
                fig4, ax4 = plt.subplots()
                ax4.plot(t_novo_acc, np.sqrt(z_acc_filtrado**2), linewidth=0.8, color='black')
                ax4.plot(A1_lat,A1_amp,'ro')
                ax4.plot(A2_lat,A2_amp,'ro')
                #ax4.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                #ax4.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8) 
                ax4.set_xlim(start_test-5,stop_test+5) 
                ax4.set_xlabel('Tempo (s)') 
                ax4.set_ylabel('Aceleração AP (m/s²)') 
                ax4.legend() 
                st.pyplot(fig4) 
            with col2: 
                fig5, ax5 = plt.subplots() 
                ax5.plot(t_novo_gyro, norma_gyro_filtrado, linewidth=0.8, color='black') 
                ax5.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8) 
                #ax5.axvline(A1v_lat, color='blue', linestyle='--', label='A1 v', linewidth=0.8) 
                #ax5.axvline(A1_lat, color='orange', linestyle='--', label='A1 AP', linewidth=0.8) 
                #ax5.axvline(G1_lat, color='black', linestyle='--', label='G1', linewidth=0.8) 
                #ax5.axvline(G2_lat, color='black', linestyle='--', label='G2', linewidth=0.8) 
                #ax5.axvline(G4_lat, color='cyan', linestyle='--', label='G4', linewidth=0.8) 
                #ax5.axvline(A2v_lat, color='yellow', linestyle='--', label='A2 v', linewidth=0.8) 
                #ax5.axvline(A2_lat, color='gray', linestyle='--', label='A2 AP', linewidth=0.8) 
                ax5.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax5.set_xlim(start_test-5,stop_test+5)
                ax5.set_xlabel('Tempo (s)')
                ax5.set_ylabel('Velocidade angular norma (rad/s)')
                ax5.legend() 
                st.pyplot(fig5)
                fig6, ax6 = plt.subplots()
                ax6.plot(t_novo_gyro, np.sqrt(v_gyro**2), linewidth=0.8, color='black')
                ax6.plot(G1_lat,G1_amp,'ro')
                ax6.plot(G2_lat,G2_amp,'ro')
                ax6.set_xlim(start_test-5,stop_test+5) 
                ax6.set_xlabel('Tempo (s)')
                ax6.set_ylabel('Velocidade angular Vertical (rad/s)')
                ax6.legend() 
                st.pyplot(fig6)
                fig7, ax7 = plt.subplots()
                ax7.plot(t_novo_gyro, np.sqrt(ml_gyro**2), linewidth=0.8, color='black')
                ax7.plot(G0_lat,G0_amp,'ro')
                ax7.plot(G4_lat,G4_amp,'ro')
                ax7.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                ax7.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax7.set_xlim(start_test-5,stop_test+5)
                ax7.set_xlabel('Tempo (s)')
                ax7.set_ylabel('Velocidade angular ML (rad/s)')
                ax7.legend()
                st.pyplot(fig7)
                fig8, ax8 = plt.subplots()
                ax8.plot(t_novo_gyro, np.sqrt(z_gyro_filtrado**2), linewidth=0.8, color='black')
                ax8.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                ax8.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax8.set_xlim(start_test-5,stop_test+5)
                ax8.set_xlabel('Tempo (s)')
                ax8.set_ylabel('Velocidade angular AP (rad/s)')
                ax8.legend()
                st.pyplot(fig8)
        elif tipo_teste == "Y test":
            dados = st.session_state["dados_acc_coluna"]
            dados2 = st.session_state["dados_acc_joelho"]
            tempo, ml, ap, v= ytestProcessing.processar_ytest1(dados,8)
            max_val = 20000
            col1, col2, col3 = st.columns(3)
            with col1:
                startRec = st.number_input( 'Indique o início do registro', value=0, step=1, max_value=max_val)
                c1_index = st.number_input( 'Índice C1 ', value=1000, step=1, max_value=20000)
            with col2: 
                endRec = st.number_input( 'Indique o final do registro', value=max_val, step=1, max_value=max_val)
                c2_index = st.number_input( 'Índice C2', value=1000, step=1, max_value=2000)
            with col3:
                filter = st.number_input( 'Indique o filtro passa-baixa', value=8.0, step=0.1, max_value=40.0)
                c3_index = st.number_input( 'Índice C3', value=1000, step=1, max_value=2000)
            showRec = st.checkbox('Mostrar o dado original', value=True)
            
            tempo, ml, ap, v = ytestProcessing.processar_ytest1(dados[0:len(dados)-10],filter)
            tempo_2, ml_2, ap_2, v_2= ytestProcessing.processar_ytest2(dados2[0:len(dados2)-10],filter)
            col1, col2 = st.columns(2)
            tempo_sel, ml_sel, ap_sel, v_sel = ytestProcessing.processar_ytest1( dados[startRec:endRec], filter)
            tempo_sel_2, ml_2_sel, ap_2_sel, v_2_sel = ytestProcessing.processar_ytest2( dados2[startRec:endRec], filter)
            
            picoSaltoCintura = np.max(v[0:1000])
            for index,valor in enumerate(v):
                if valor == picoSaltoCintura:
                    onsetCintura = index
                    tempo = tempo - tempo[onsetCintura]
                    break
            
            picoSaltoJoelho = np.max(v_2[0:1000])
            for index,valor in enumerate(v_2):
                if valor == picoSaltoJoelho:
                    onsetJoelho = index
                    tempo_2 = tempo_2 - tempo_2[onsetJoelho]
                    break
            
            picoSaltoCintura_sel = np.max(v_sel[0:1000])
            for index,valor in enumerate(v_sel):
                if valor == picoSaltoCintura_sel:
                    onsetCintura_sel = index
                    tempo_sel = tempo_sel - tempo_sel[onsetCintura_sel]
                    break
            picoSaltoJoelho_sel = np.max(v_2_sel[0:1000])
            for index,valor in enumerate(v_2_sel):
                if valor == picoSaltoJoelho_sel:
                    onsetJoelho_sel = index
                    tempo_sel_2 = tempo_sel_2 - tempo_sel_2[onsetJoelho_sel]
                    break
            
            ap_sel_media = uniform_filter1d(ap_sel, size=30)
            ml_sel_media = uniform_filter1d(ml_sel, size=30)
            v_sel_media = uniform_filter1d(v_sel, size=30)
            ap_2_sel_media = uniform_filter1d(ap_2_sel, size=30)
            ml_2_sel_media = uniform_filter1d(ml_2_sel, size=30)
            v_2_sel_media = uniform_filter1d(v_2_sel, size=30)
            n1 = np.max(tempo_sel)
            n2 = np.max(tempo_sel_2)
            if n1 > n2:
                limite_tempo = n1
            else: 
                limite_tempo = n2
            
            min_c1 = np.min(ap_sel_media[startRec:startRec+c1_index])
            for index,valor in enumerate(ap_sel_media):
                if valor == min_c1: 
                    t_min_c1 = tempo_sel[index]
                    break
            max_c1 = np.max(ap_sel_media[index:index+c2_index])
            for index,valor in enumerate(ap_sel_media):
                if valor == max_c1:
                    t_max_c1 = tempo_sel[index]
                    break
            #min_c2 = np.min(ap_sel_media[index:index+c3_index])
            #for index,valor in enumerate(ap_sel_media):
                #if valor == min_c2:
                    #t_min_c2 = tempo_sel[index]
                    #break
            max_c2 = np.max(ap_sel_media[index+c3_index:index+(500+c3_index)])
            for index,valor in enumerate(ap_sel_media):
                if valor == max_c2:
                    t_max_c2 = tempo_sel[index]
                    break
            with col1:
                st.title("Coluna vertebral") 
                # Cria figura com GridSpec personalizado
                # Cria uma figura com 3 subplots verticais 
                fig = plt.figure(figsize=(12, 10))
                gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.3, hspace=0.6) 
                limite = 5 
                ax1 = fig.add_subplot(gs[0, 0])
                if showRec:
                    ax1.plot( tempo_sel[startRec:endRec], ap_sel[startRec:endRec], color='black', linewidth=0.8) 
                ax1.plot( tempo_sel[startRec:endRec], ap_sel_media[startRec:endRec], color='red', linewidth=0.8)
                ax1.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                ax1.plot([t_max_c1,t_max_c1],[-4,4],"--b")
                #ax1.plot([t_min_c2,t_min_c2],[-4,4],"--g")
                ax1.plot([t_max_c2,t_max_c2],[-4,4],"--k")
                ax1.set_xlabel(r'Tempo (s)', fontsize=8)
                ax1.set_ylabel(r'Aceleração AP (m/s$^2$)', fontsize=8)
                ax1.set_xlim(tempo_sel[startRec], tempo_sel[-1])
                #ax1.set_ylim(-limite, limite)
                ax1.tick_params(axis='both', labelsize=8)
                # Gráfico 2: ocupa linha superior direita (metade superior)
                ax2 = fig.add_subplot(gs[1, 0])
                if showRec:
                    ax2.plot( tempo_sel[startRec:endRec], ml_sel[startRec:endRec], color='black', linewidth=0.8)
                ax2.plot(tempo_sel[startRec:endRec], ml_sel_media[startRec:endRec], color='red', linewidth=0.8)
                ax2.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                ax2.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                #ax2.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                ax2.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                ax2.set_xlabel('Tempo (s)', fontsize=8)
                ax2.set_ylabel(r'Aceleração ML (m/s$^2$)', fontsize=8)
                ax2.set_xlim(tempo_sel[startRec], tempo_sel[-1])
                ax2.set_ylim(-limite, limite)
                ax2.tick_params(axis='both', labelsize=8)
                # Gráfico 2: ocupa linha superior direita (metade superior) 
                axv = fig.add_subplot(gs[2, 0])
                if showRec:
                    axv.plot(tempo_sel[startRec:endRec], v_sel[startRec:endRec], color='black', linewidth=0.8)
                axv.plot(tempo_sel[startRec:endRec], v_sel_media[startRec:endRec], color='red', linewidth=0.8)
                axv.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                axv.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                #axv.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                axv.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                axv.set_xlabel('Tempo (s)', fontsize=8)
                axv.set_ylabel(r'Aceleração V (m/s$^2$)', fontsize=8)
                axv.set_xlim(tempo_sel[startRec], tempo_sel[-1])
                axv.set_ylim(-limite, limite)
                axv.tick_params(axis='both', labelsize=8) 
                # Exibe no Streamlit 
                st.pyplot(fig)
            with col2: 
                st.title("Joelho")
                # Cria figura com GridSpec personalizado
                # Cria uma figura com 3 subplots verticais
                fig_2 = plt.figure(figsize=(12, 10))
                gs_2 = gridspec.GridSpec(3, 2, figure=fig_2, wspace=0.3, hspace=0.6)
                ax1_2 = fig_2.add_subplot(gs_2[0, 0])
                if showRec:
                    ax1_2.plot( tempo_sel_2[startRec:endRec], ap_2_sel[startRec:endRec], color='black', linewidth=0.8)
                ax1_2.plot( tempo_sel_2[startRec:endRec], ap_2_sel_media[startRec:endRec], color='blue', linewidth=0.8)
                ax1_2.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                ax1_2.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                #ax1_2.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                ax1_2.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                ax1_2.set_xlabel(r'Tempo (s)', fontsize=8)
                ax1_2.set_ylabel(r'Aceleração AP (m/s$^2$)', fontsize=8)
                ax1_2.set_xlim(tempo_sel_2[startRec], tempo_sel_2[-1])
                ax1_2.set_ylim(-limite, limite)
                ax1_2.tick_params(axis='both', labelsize=8) 
                # Gráfico 2: ocupa linha superior direita (metade superior) 
                ax2_2 = fig_2.add_subplot(gs_2[1, 0])
                if showRec:
                    ax2_2.plot( tempo_sel_2[startRec:endRec], ml_2_sel[startRec:endRec], color='black', linewidth=0.8)
                ax2_2.plot(tempo_sel_2[startRec:endRec], ml_2_sel_media[startRec:endRec], color='blue', linewidth=0.8)
                ax2_2.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                ax2_2.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                #ax2_2.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                ax2_2.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                ax2_2.set_xlabel('Tempo (s)', fontsize=8)
                ax2_2.set_ylabel(r'Aceleração ML (m/s$^2$)', fontsize=8)
                ax2_2.set_xlim(tempo_sel_2[startRec], tempo_sel_2[-1])
                ax2_2.set_ylim(-limite, limite)
                ax2_2.tick_params(axis='both', labelsize=8)
                # Gráfico 2: ocupa linha superior direita (metade superior)
                axv_2 = fig_2.add_subplot(gs_2[2, 0])
                if showRec:
                    axv_2.plot(tempo_sel_2[startRec:endRec], v_2_sel[startRec:endRec], color='black', linewidth=0.8)
                axv_2.plot(tempo_sel_2[startRec:endRec], v_2_sel_media[startRec:endRec], color='blue', linewidth=0.8)
                axv_2.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                axv_2.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                #axv_2.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                axv_2.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                axv_2.set_xlabel('Tempo (s)', fontsize=8)
                axv_2.set_ylabel(r'Aceleração V (m/s$^2$)', fontsize=8)
                axv_2.set_xlim(tempo_sel_2[startRec], tempo_sel_2[-1])
                axv_2.set_ylim(-limite, limite)
                axv_2.tick_params(axis='both', labelsize=8) 
                # Exibe no Streamlit 
                st.pyplot(fig_2)

        elif tipo_teste == "Propriocepção":
            calibracao = st.number_input('Indique o valor angular da extensão do cotovelo (em graus)', value=0.0)
            t1 = st.number_input('Indique o limite inferior da janela temporal de análise (s)', value=0.0)
            t2 = st.number_input('Indique o limite superior da janela temporal de análise (s)', value=t1+10.0)
            dados = st.session_state["dados"]
            st.session_state["calibracao"] = calibracao
            tempo, x_vf, y_vf, z_vf = jointSenseProcessing.processar_jps(dados, 8)
            max_val = len(tempo) # Cálculo dos ângulos
            accelAngleX = np.arctan(y_vf / np.sqrt(x_vf**2 + z_vf**2)) * 180 / math.pi
            angulo = accelAngleX + 90 # Calibração 
            def objetivo(x): 
                media_ajustada = np.mean(angulo[100:500] + x) 
                return abs(media_ajustada - calibracao) 
            media_baseline = np.mean(angulo[100:500])
            if media_baseline != calibracao:
                resultado = minimize_scalar(objetivo)
                angulo = angulo + resultado.x 
            else: 
                resultado = type('obj', (object,), {'x': 0})() 
            for index,valor in enumerate(tempo): 
                if valor > t1: 
                    index_t1 = index
                    break
            for index,valor in enumerate(tempo): 
                if valor > t2: 
                    index_t2 = index
                    break
            ref_max = np.max(angulo[index_t1:index_t2]) 
           
            col1,col2,col3 = st.columns([0.2,0.8,0.2])# Cria figura com GridSpec personalizado 
            with col2:
                fig = plt.figure(figsize=(8, 10))
                gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.8, wspace=0.6)
                ax1 = fig.add_subplot(gs[0:2, 0:2])
                ax1.plot(tempo, angulo, color='tomato', linewidth=0.5)
                ax1.plot([tempo[index_t1],tempo[index_t1]],[0,120],'k--')
                ax1.plot([tempo[index_t2],tempo[index_t2]],[0,120],'k--')
                st.pyplot(fig)
        else:
            st.info("Dados ou tipo de teste não definidos. Vá até a aba 'Importar Dados'.")

# === Página de Exportação ===
elif pagina == "📤 Exportar Resultados":
    if "dados" not in st.session_state or "tipo_teste" not in st.session_state:
        st.info("Dados ou tipo de teste não definidos. Vá até a aba 'Importar Dados'.")
    else:
        tipo_teste = st.session_state["tipo_teste"]
        st.subheader(f"📤 Exportação - {tipo_teste}")

        if tipo_teste == "Equilíbrio":
            if "intervalo" not in st.session_state:
                st.warning("Defina o intervalo na página de Visualização primeiro.")
            else:
                startRec, endRec, fc = st.session_state["intervalo"]
                modo_equilibrio = st.session_state.get("equilibrio_modo_processamento", "individual")

                if modo_equilibrio == "3_arquivos":
                    dados_lista = st.session_state.get("equilibrio_dados_3")
                    if not dados_lista or len(dados_lista) != 3:
                        st.error("Dados do Equilíbrio (3 arquivos) não encontrados. Reimporte os arquivos.")
                    else:
                        resultados_individuais = [
                            _calcular_variaveis_equilibrio(dados, startRec, endRec, fc)
                            for dados in dados_lista
                        ]
                        st.session_state["equilibrio_resultados_individuais"] = resultados_individuais
                        medias_df = _montar_df_medias_equilibrio(resultados_individuais)
                        st.session_state["equilibrio_resultado_medias_df"] = medias_df

                        st.download_button(
                            "📄 Exportar médias (CSV)",
                            data=medias_df.to_csv(index=False),
                            file_name="equilibrio_medias_3_arquivos.csv",
                            mime="text/csv"
                        )

                        for indice, variaveis in enumerate(resultados_individuais, start=1):
                            resultado_txt = _formatar_resultado_txt(variaveis)
                            st.download_button(
                                f"📄 Exportar resultados execução {indice} (.txt)",
                                data=resultado_txt,
                                file_name=f"resultados_analise_postural_execucao_{indice}.txt",
                                mime="text/plain"
                            )
                else:
                    dados = st.session_state["dados"]
                    variaveis = _calcular_variaveis_equilibrio(dados, startRec, endRec, fc)
                    resultado_txt = _formatar_resultado_txt(variaveis)

                    st.download_button(
                        "📄 Exportar resultados (.txt)",
                        data=resultado_txt,
                        file_name="resultados_analise_postural.txt",
                        mime="text/plain"
                    )
        if tipo_teste == "TUG":
            
            baseline_onset = st.session_state["baseline_onset"]
            baseline_offset = st.session_state["baseline_offset"]
            dados_acc = st.session_state["dados_acc"]
            dados_gyro = st.session_state["dados_gyro"]
            baseline_onset = st.number_input('Indique o momento inicial da baseline do início do teste (s)', value=0.0)
            baseline_offset = st.number_input('Indique o momento inicial da baseline do final do teste (s)', value=(np.max(dados_acc["Tempo"])/1000)-2.5)
            col1, col2, col3 = st.columns([0.4,0.8,0.6])
            t_novo_acc, v_acc, ml_acc, z_acc_filtrado, norma_acc_filtrado, t_novo_gyro, v_gyro, ml_gyro, z_gyro_filtrado, norma_gyro_filtrado,start_test,stop_test,idx,idx_ml,idx_acc_ap,idx_acc_v,duration = tugProcessing.processar_tug(dados_acc,dados_gyro,2,1.25,baseline_onset,baseline_offset) 
            vertical_squared = np.sqrt(v_gyro**2) 
            lat1 = idx[1][0] 
            lat2 = idx[1][1] 
            amp1 = vertical_squared[idx[0][0]] 
            amp2 = vertical_squared[idx[0][1]] 
            if lat1 > lat2: 
                G1_lat = lat2 
                G1_amp = amp2 
                G2_lat = lat1 
                G2_amp = amp1 
            else: 
                G1_lat = lat1 
                G1_amp = amp1 
                G2_lat = lat2 
                G2_amp = amp2 
            ml_squared = np.sqrt(ml_gyro**2) 
            lat1 = idx_ml[1][0] 
            lat2 = idx_ml[1][1] 
            amp1 = ml_squared[idx_ml[0][0]] 
            amp2 = ml_squared[idx_ml[0][1]] 
            if lat1 > lat2: 
                G0_lat = lat2 
                G0_amp = amp2 
                G4_lat = lat1 
                G4_amp = amp1 
            else: 
                G0_lat = lat1 
                G0_amp = amp1 
                G4_lat = lat2 
                G4_amp = amp2 
            acc_ap_squared = np.sqrt(z_acc_filtrado**2) 
            lat1 = idx_acc_ap[1][0] 
            lat2 = idx_acc_ap[1][1] 
            amp1 = acc_ap_squared[idx_acc_ap[0][0]] 
            amp2 = acc_ap_squared[idx_acc_ap[0][1]] 
            if lat1 > lat2: 
                A1_lat = lat2 
                A1_amp = amp2 
                A2_lat = lat1 
                A2_amp = amp1 
            else: 
                A1_lat = lat1 
                A1_amp = amp1 
                A2_lat = lat2 
                A2_amp = amp2 
            acc_v_squared = np.sqrt(v_acc**2) 
            lat1 = idx_acc_v[1][0] 
            lat2 = idx_acc_v[1][1] 
            amp1 = acc_v_squared[idx_acc_v[0][0]] 
            amp2 = acc_v_squared[idx_acc_v[0][1]] 
            if lat1 > lat2: 
                A1v_lat = lat2 
                A1v_amp = amp2 
                A2v_lat = lat1 
                A2v_amp = amp1 
            else: 
                A1v_lat = lat1
                A1v_amp = amp1 
                A2v_lat = lat2 
                A2v_amp = amp2 
            with col1: 
                st.metric(label=r"Duração do teste (s)", value=round(stop_test-start_test, 4))
                st.metric(label=r"Duração para levantar (s)", value=round(G0_lat-start_test, 4)) 
                st.metric(label=r"Duração da caminhada de ida (s)", value=round(G1_lat-G0_lat, 4)) 
                st.metric(label=r"Duração da caminhada de volta (s)", value=round(G2_lat-G1_lat, 4)) 
                st.metric(label=r"Duração para sentar (s)", value=round(stop_test-G2_lat, 4)) 
            with col2: 
                st.metric(label=r"Tempo para o pico de velocidade angular na transição de sentado para de pé (s)", value=round(G0_lat-start_test, 4)) 
                st.metric(label=r"Tempo para o pico de aceleração AP na transição de sentado para de pé (s)", value=round(A1_lat-start_test, 4)) 
                st.metric(label=r"Tempo para o pico de aceleração V na transição de sentado para de pé (s)", value=round(A1v_lat-start_test, 4)) 
                st.metric(label=r"Tempo para o pico de velocidade angular no giro em 3 m (s)", value=round(G1_lat-start_test, 4)) 
                st.metric(label=r"Tempo para o pico de velocidade angular no giro em 6 m (s)", value=round(G2_lat-start_test, 4)) 
                st.metric(label=r"Tempo para o pico de velocidade angular na transição de pé para sentado (s)", value=round(G2_lat-start_test, 4)) 
                st.metric(label=r"Tempo para o pico de aceleração AP na transição de pé para sentado (s)", value=round(A2_lat-start_test, 4)) 
                st.metric(label=r"Tempo para o pico de aceleração V na transição de pé para sentado (s)", value=round(A2v_lat-start_test, 4)) 
            with col3: 
                st.metric(label=r"Velocidade angular máxima na transição de sentado para de pé (rad/s)", value=round(G0_amp, 4)) 
                st.metric(label=r"Aceleração máxima AP na transição de sentado para de pé (m/s2)", value=round(A1_amp, 4)) 
                st.metric(label=r"Aceleração máxima V na transição de sentado para de pé (m/s2)", value=round(A1v_amp, 4)) 
                st.metric(label=r"Velocidade angular máxima no giro em 3 m (s)", value=round(G1_amp, 4)) 
                st.metric(label=r"Velocidade angular máxima no giro em 6 m (s)", value=round(G2_amp, 4)) 
                st.metric(label=r"Aceleração máxima AP na transição de pé para sentado (s)", value=round(A2_amp, 4)) 
                st.metric(label=r"Aceleração máxima V na transição de pé para sentado (s)", value=round(A2v_amp, 4)) 
                
            resultado_txt = "Variável\tValor\n" # Cabeçalho com tabulação # Lista de pares (nome, valor) 
            variaveis = [("Duração do teste (s)", round(stop_test-start_test, 4)), ("Duração para levantar (s)", round(G0_lat-start_test, 4)), ("Duração da caminhada de ida (s)",round(G1_lat-G0_lat, 4)), ("Duração da caminhada de volta (s)",round(G2_lat-G1_lat, 4)), ("Duração para sentar (s)",round(stop_test-G2_lat, 4)), ("Tempo para o pico de velocidade angular na transição de sentado para de pé (s)",round(G0_lat-start_test, 4)), ("Tempo para o pico de aceleração AP na transição de sentado para de pé (s)",round(A1_lat-start_test, 4)), ("Tempo para o pico de aceleração V na transição de sentado para de pé (s)",round(A1v_lat-start_test, 4)), ("Tempo para o pico de velocidade angular no giro em 3 m (s)",round(G1_lat-start_test, 4)), ("Tempo para o pico de velocidade angular no giro em 6 m (s)",round(G2_lat-start_test, 4)), ("Tempo para o pico de velocidade angular na transição de pé para sentado (s)",round(A2_lat-start_test, 4)), ("Tempo para o pico de aceleração AP na transição de pé para sentado (s)",round(A2_lat-start_test, 4)), ("Tempo para o pico de aceleração V na transição de pé para sentado (s)",round(A2v_lat-start_test, 4)), ("Velocidade angular máxima na transição de sentado para de pé (rad/s)",round(G0_amp, 4)), ("Aceleração máxima AP na transição de sentado para de pé (m/s2)",round(A1_amp, 4)), ("Aceleração máxima V na transição de sentado para de pé (m/s2)",round(A1v_amp, 4)), ("Velocidade angular máxima no giro em 3 m (s)",round(G1_amp, 4)), ("Velocidade angular máxima no giro em 6 m (s)",round(G2_amp, 4)), ("Aceleração máxima AP na transição de sentado para pé (s)",round(A2_amp, 4)), ("Aceleração máxima V na transição de sentado para pé (s)",round(A2v_amp, 4))]
            # Adiciona linha por linha 
            
            for nome, valor in variaveis: 
                resultado_txt += f"{nome}\t{valor}\n" 
            st.download_button(label="📄 Exportar resultados (.txt)", data=resultado_txt, file_name="resultados_analise_iTUG.txt", mime="text/plain" ) 
        if tipo_teste == "Y test": 
            dados = st.session_state["dados_acc_coluna"]
            dados2 = st.session_state["dados_acc_joelho"]
            tempo, ml, ap, v= ytestProcessing.processar_ytest1(dados,8)
            max_val = 20000
            col1, col2, col3 = st.columns(3)
            with col1:
                startRec = st.number_input( 'Indique o início do registro', value=0, step=1, max_value=max_val)
                c1_index = st.number_input( 'Índice C1 ', value=1000, step=1, max_value=2000)
            with col2: 
                endRec = st.number_input( 'Indique o final do registro', value=max_val, step=1, max_value=max_val)
                c2_index = st.number_input( 'Índice C2', value=1000, step=1, max_value=2000)
            with col3:
                filter = st.number_input( 'Indique o filtro passa-baixa', value=8.0, step=0.1, max_value=40.0)
                c3_index = st.number_input( 'Índice C3', value=1000, step=1, max_value=2000)
            showRec = st.checkbox('Mostrar o dado original', value=True)
            
            tempo, ml, ap, v = ytestProcessing.processar_ytest1(dados[0:len(dados)-10],filter)
            tempo_2, ml_2, ap_2, v_2= ytestProcessing.processar_ytest2(dados2[0:len(dados2)-10],filter)
            col1, col2 = st.columns(2)
            tempo_sel, ml_sel, ap_sel, v_sel = ytestProcessing.processar_ytest1( dados[startRec:endRec], filter)
            tempo_sel_2, ml_2_sel, ap_2_sel, v_2_sel = ytestProcessing.processar_ytest2( dados2[startRec:endRec], filter)
            picoSaltoCintura = np.max(v[0:1000])
            for index,valor in enumerate(v):
                if valor == picoSaltoCintura:
                    onsetCintura = index
                    tempo = tempo - tempo[onsetCintura]
                    break
            picoSaltoJoelho = np.max(v_2[0:1000])
            for index,valor in enumerate(v_2):
                if valor == picoSaltoJoelho:
                    onsetJoelho = index
                    tempo_2 = tempo_2 - tempo_2[onsetJoelho]
                    break
            picoSaltoCintura_sel = np.max(v_sel[0:1000])
            for index,valor in enumerate(v_sel):
                if valor == picoSaltoCintura_sel:
                    onsetCintura_sel = index
                    tempo_sel = tempo_sel - tempo_sel[onsetCintura_sel]
                    break
            picoSaltoJoelho_sel = np.max(v_2_sel[0:1000])
            for index,valor in enumerate(v_2_sel):
                if valor == picoSaltoJoelho_sel:
                    onsetJoelho_sel = index
                    tempo_sel_2 = tempo_sel_2 - tempo_sel_2[onsetJoelho_sel]
                    break
            ap_sel_media = uniform_filter1d(ap_sel, size=30)
            ml_sel_media = uniform_filter1d(ml_sel, size=30)
            v_sel_media = uniform_filter1d(v_sel, size=30)
            ap_2_sel_media = uniform_filter1d(ap_2_sel, size=30)
            ml_2_sel_media = uniform_filter1d(ml_2_sel, size=30)
            v_2_sel_media = uniform_filter1d(v_2_sel, size=30)
            n1 = np.max(tempo_sel)
            n2 = np.max(tempo_sel_2)
            if n1 > n2:
                limite_tempo = n1
            else: 
                limite_tempo = n2
            
            min_c1 = np.min(ap_sel_media[startRec:startRec+c1_index])
            for index,valor in enumerate(ap_sel_media):
                if valor == min_c1: 
                    t_min_c1 = tempo_sel[index]
                    break
            for index2,valor in enumerate(tempo_sel_2):
                if valor > t_min_c1: 
                    break 
            dados = ap_2_sel_media[index2-50:index2] 
            ampC1_ap_pre = np.sqrt(np.mean(np.square(dados))) 
            dados = ap_2_sel_media[index2:index2+50] 
            ampC1_ap_pos = np.sqrt(np.mean(np.square(dados))) 
            dados = ml_2_sel_media[index2-50:index2] 
            ampC1_ml_pre = np.sqrt(np.mean(np.square(dados))) 
            dados = ml_2_sel_media[index2:index2+50] 
            ampC1_ml_pos = np.sqrt(np.mean(np.square(dados))) 

            max_c1 = np.max(ap_sel_media[index:index+c2_index])
            for index,valor in enumerate(ap_sel_media):
                if valor == max_c1:
                    t_max_c1 = tempo_sel[index]
                    break
            for index3,valor in enumerate(tempo_sel_2):
                if valor > t_max_c1:
                    break 
            dados = ap_2_sel_media[index3-50:index3]
            ampC2_ap_pre = np.sqrt(np.mean(np.square(dados)))
            dados = ap_2_sel_media[index3:index3+50]
            ampC2_ap_pos = np.sqrt(np.mean(np.square(dados)))
            dados = ml_2_sel_media[index3-50:index3]
            ampC2_ml_pre = np.sqrt(np.mean(np.square(dados)))
            dados = ml_2_sel_media[index3:index3+50]
            ampC2_ml_pos = np.sqrt(np.mean(np.square(dados)))

            max_c2 = np.max(ap_sel_media[index+c3_index:index+(500+c3_index)])
            for index,valor in enumerate(ap_sel_media):
                if valor == max_c2:
                    t_max_c2 = tempo_sel[index]
                    break
            for index5,valor in enumerate(tempo_sel_2):
                if valor > t_max_c2: 
                    break 
            dados = ap_2_sel_media[index5-50:index5] 
            ampC4_ap_pre = np.sqrt(np.mean(np.square(dados))) 
            dados = ap_2_sel_media[index5:index5+50] 
            ampC4_ap_pos = np.sqrt(np.mean(np.square(dados))) 
            dados = ml_2_sel_media[index5-50:index5] 
            ampC4_ml_pre = np.sqrt(np.mean(np.square(dados))) 
            dados = ml_2_sel_media[index5:index5+50] 
            ampC4_ml_pos = np.sqrt(np.mean(np.square(dados))) 
            col1,col2,col3,col4 = st.columns(4) 
            with col1: 
                st.metric(label=r"Amplitude de C1 (m/s2)", value=round(min_c1, 4)) 
                st.metric(label=r"Tempo de C1 (s)", value=round(t_min_c1, 4)) 
                st.metric(label=r"Amplitude pré-C1 Joelho AP (m/s2)", value=round(ampC1_ap_pre, 4)) 
                st.metric(label=r"Amplitude pós-C1 Joelho AP (m/s2)", value=round(ampC1_ap_pos, 4)) 
                st.metric(label=r"Amplitude pré-C1 Joelho ML (m/s2)", value=round(ampC1_ml_pre, 4)) 
                st.metric(label=r"Amplitude pós-C1 Joelho ML (m/s2)", value=round(ampC1_ml_pos, 4)) 
            with col2: 
                st.metric(label=r"Amplitude de C2 (m/s2)", value=round(max_c1, 4)) 
                st.metric(label=r"Tempo de C2 (s)", value=round(t_max_c1, 4)) 
                st.metric(label=r"Amplitude pré-C2 Joelho AP (m/s2)", value=round(ampC2_ap_pre, 4)) 
                st.metric(label=r"Amplitude pós-C2 Joelho AP (m/s2)", value=round(ampC2_ap_pos, 4)) 
                st.metric(label=r"Amplitude pré-C2 Joelho ML (m/s2)", value=round(ampC2_ml_pre, 4)) 
                st.metric(label=r"Amplitude pós-C2 Joelho ML (m/s2)", value=round(ampC2_ml_pos, 4))  
                
            with col4: 
                st.metric(label=r"Amplitude de C3 (m/s2)", value=round(max_c2, 4)) 
                st.metric(label=r"Tempo de C3 (s)", value=round(t_max_c2, 4)) 
                st.metric(label=r"Amplitude pré-C3 Joelho AP (m/s2)", value=round(ampC4_ap_pre, 4)) 
                st.metric(label=r"Amplitude pós-C3 Joelho AP (m/s2)", value=round(ampC4_ap_pos, 4)) 
                st.metric(label=r"Amplitude pré-C3 Joelho ML (m/s2)", value=round(ampC4_ml_pre, 4)) 
                st.metric(label=r"Amplitude pós-C3 Joelho ML (m/s2)", value=round(ampC4_ml_pos, 4)) 

            variaveis = [("Amplitude de C1 (m/s2)", round(min_c1, 4)), ("Tempo de C1 (s)", round(t_min_c1, 4)), ("Amplitude de C2 (m/s2)", round(max_c1, 4)), ("Tempo de C2 (s)", round(t_max_c1, 4)), ("Amplitude de C3 (m/s2)", round(max_c2, 4)), ("Tempo de C3 (s)", round(t_max_c2, 4)), ("Amplitude pré-C1 Joelho AP (m/s2)",round(ampC1_ap_pre, 4)), ("Amplitude pós-C1 Joelho AP (m/s2)",round(ampC1_ap_pos, 4)), ("Amplitude pré-C2 Joelho AP (m/s2)",round(ampC2_ap_pre, 4)), ("Amplitude pós-C2 Joelho AP (m/s2)",round(ampC2_ap_pos, 4)), ("Amplitude pré-C3 Joelho AP (m/s2)",round(ampC4_ap_pre, 4)), ("Amplitude pós-C3 Joelho AP (m/s2)",round(ampC4_ap_pos, 4)), ("Amplitude pré-C1 Joelho ML (m/s2)",round(ampC1_ml_pre, 4)), ("Amplitude pós-C1 Joelho ML (m/s2)",round(ampC1_ml_pos, 4)), ("Amplitude pré-C2 Joelho ML (m/s2)",round(ampC2_ml_pre, 4)), ("Amplitude pós-C2 Joelho ML (m/s2)",round(ampC2_ml_pos, 4)), ("Amplitude pré-C3 Joelho ML (m/s2)",round(ampC4_ml_pre, 4)), ("Amplitude pós-C3 Joelho ML (m/s2)",round(ampC4_ml_pos, 4)),] 
            resultado_txt = "Variável\tValor\n" # Cabeçalho com tabulação 
            for nome, valor in variaveis: 
                resultado_txt += f"{nome}\t{valor}\n" 
            st.download_button( label="📄 Exportar resultados (.txt)", data=resultado_txt, file_name="resultados_testY.txt", mime="text/plain" )

            def _serie_para_txt(t, y, col_t="Tempo_s", col_y="Acc_AP_m_s2", sep="\t"):
                df_out = pd.DataFrame({
                    col_t: np.asarray(t, dtype=float),
                    col_y: np.asarray(y, dtype=float),
                })
                buf = io.StringIO()
                df_out.to_csv(buf, sep=sep, index=False, float_format="%.6f")
                return buf.getvalue()
            
            # 1) Série bruta/filtrada (ap_sel)
            txt_ap_cintura = _serie_para_txt(
                tempo_sel[startRec:endRec],
                ap_sel_media[startRec:endRec],
                col_t="Tempo_s",
                col_y="Acc_AP_Cintura_m_s2",
                sep="\t"
            )
            
            st.download_button(
                label="📄 Exportar Tempo + Aceleração AP (cintura) (.txt)",
                data=txt_ap_cintura,
                file_name="ytest_cintura_tempo_ap.txt",
                mime="text/plain"
            )
            # 1) Série bruta/filtrada (ap_sel)
            txt_ap_joelho = _serie_para_txt(
                tempo_sel_2[startRec:endRec],
                ap_2_sel_media[startRec:endRec],
                col_t="Tempo_s",
                col_y="Acc_AP_Joelho_m_s2",
                sep="\t"
            )
            
            st.download_button(
                label="📄 Exportar Tempo + Aceleração AP (Joelho) (.txt)",
                data=txt_ap_joelho,
                file_name="ytest_joelho_ap_tempo.txt",
                mime="text/plain"
            )

            # 1) Série bruta/filtrada (ap_sel)
            txt_ml_joelho = _serie_para_txt(
                tempo_sel_2[startRec:endRec],
                ml_2_sel_media[startRec:endRec],
                col_t="Tempo_s",
                col_y="Acc_ML_Joelho_m_s2",
                sep="\t"
            )
            
            st.download_button(
                label="📄 Exportar Tempo + Aceleração ML (Joelho) (.txt)",
                data=txt_ml_joelho,
                file_name="ytest_joelho_ml_tempo_ap.txt",
                mime="text/plain"
            )

# (Opcional) 2) Série suavizada (ap_sel_media), se você quiser também
#txt_ap_cintura_suave = _serie_para_txt(
#    tempo_sel[startRec:endRec],
#    ap_sel_media[startRec:endRec],
#    col_t="Tempo_s",
#    col_y="Acc_AP_Cintura_suavizada_m_s2",
#    sep="\t"
#)

#st.download_button(
#    label="📄 Exportar Tempo + AP suavizada (cintura) (.txt)",
#    data=txt_ap_cintura_suave,
#    file_name="ytest_cintura_tempo_ap_suavizada.txt",
#    mime="text/plain"
#)
        if tipo_teste == "Propriocepção": 
            calibracao = st.number_input('Indique o valor angular da extensão do cotovelo (em graus)', value=0.0)
            t1 = st.number_input('Indique o limite inferior da janela temporal de análise (s)', value=0.0)
            t2 = st.number_input('Indique o limite superior da janela temporal de análise (s)', value=t1+10.0)
            dados = st.session_state["dados"]
            st.session_state["calibracao"] = calibracao
            tempo, x_vf, y_vf, z_vf = jointSenseProcessing.processar_jps(dados, 8)
            max_val = len(tempo) # Cálculo dos ângulos
            accelAngleX = np.arctan(y_vf / np.sqrt(x_vf**2 + z_vf**2)) * 180 / math.pi
            angulo = accelAngleX + 90 # Calibração 
            def objetivo(x): 
                media_ajustada = np.mean(angulo[100:500] + x) 
                return abs(media_ajustada - calibracao) 
            media_baseline = np.mean(angulo[100:500])
            if media_baseline != calibracao:
                resultado = minimize_scalar(objetivo)
                angulo = angulo + resultado.x 
            else: 
                resultado = type('obj', (object,), {'x': 0})() 
            for index,valor in enumerate(tempo): 
                if valor > t1: 
                    index_t1 = index
                    break
            for index,valor in enumerate(tempo): 
                if valor > t2: 
                    index_t2 = index
                    break
            ref_max = np.mean(angulo[index_t1:index_t2]) 
            st.metric(label=r"Média angular no intervalo (graus)", value=round(ref_max, 4)) 
            
            resultado_txt = "Variável\tValor\n" # Cabeçalho com tabulação 
            variaveis = [("Média angular no intervalo (graus)", round(ref_max, 4))] 
            for nome, valor in variaveis: 
                resultado_txt += f"{nome}\t{valor}\n" 
                st.download_button( label="📄 Exportar resultados (.txt)", data=resultado_txt, file_name="resultados_propriocepcao.txt", mime="text/plain" )
           
            
        else:
            st.info("Dados ou tipo de teste não definidos. Vá até a aba 'Importar Dados'.")
            
            
            
            
            
            
            
            
            
#elif pagina == "📖 Referências bibliográficas": 
    #html = dedent(""" 
    #<div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;"> 
    #<p> Artigos que utilizaram aplicativos desenvolvidos no projeto Momentum: 
    #</p> <a href="https://www.mdpi.com/1424-8220/24/9/2918" target="_blank" style="color:#1E90FF; text-decoration:none;">1. SANTOS, P. S. A. ; SANTOS, E. G. R. ; MONTEIRO, L. C. P. ; SANTOS-LOBATO, B. L. ; PINTO, G. H. L. ; BELGAMO, A. ; ANDRÉ DOS SANTOS, CABRAL ; COSTA E SILVA, A. A ; CALLEGARI, B. ; SOUZA, Givago da Silva . The hand tremor spectrum is modified by the inertial sensor mass during lightweight wearable and smartphone-based assessment in healthy young subjects. Scientific Reports, v. 12, p. 01, 2022.</a></p>. <a href="https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2023.1277408/full" target="_blank" style="color:#1E90FF; text-decoration:none;">2. RODRIGUES, L. A. ; SANTOS, E. G. R. ; SANTOS, P. S. A. ; IGARASHI, Y. ; OLIVEIRA, L. K. R. ; PINTO, G. H. L. ; SANTOS-LOBATO, B. L. ; CABRAL, A. S. ; BELGAMO, A. ; COSTA E SILVA, A. A ; CALLEGARI, B. ; Souza, G. S. . Wearable Devices and Smartphone Inertial Sensors for Static Balance Assessment: A Concurrent Validity Study in Young Adult Population. Journal Of Personalized Medicine, v. 1, p. 1-1, 2022.</a></p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">3. CORREA, B. D. C. ; SANTOS, E. G. R. ; BELGAMO, A. ; PINTO, G. H. L. ; XAVIER, S. S. ; DIAS, A. R. N. ; PARANHOS, A. C. M. ; ANDRÉ DOS SANTOS, CABRAL ; CALLEGARI, B. ; COSTA E SILVA, A. A. ; QUARESMA, J. A. S. ; FALCAO, L. F. M. ; SOUZA, GIVAGO S. . SMARTPHONE-BASED EVALUATION OF STATIC BALANCE AND MOBILITY IN LONG LASTING COVID-19 PATIENTS. Frontiers in Neurology, v. 1, p. 1, 2023.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">4. FURTADO, E. C. S. ; AZEVEDO, Y. S. ; GALHARDO, D. R. ; MIRANDA, I. P. C. ; OLIVEIRA, M. E. C. ; NEVES, P. F. M. ; MONTE, L. B. ; NUNES, E. F. C. ; FERREIRA, E. A. G. ; CALLEGARI, B. ; SOUZA, G.S. ; MELO NETO, J. S. . The weeks of gestation age influence the myoelectric activity of the pelvic floor muscles, plantar contact and functional mobility in high-risk pregnant women? a cross-sectional study. SENSORS, v. 1, p. 1, 2024.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">5. SANTOS, T. T. S. ; MARQUES, A. P. ; MONTEIRO, L. C. P. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; BELGAMO, A. ; COSTA E SILVA, A. A. ; CABRAL, A. S. ; KULIŚ ; GAJEWSKI, J. ; Souza, G. S. ; SILVA, T. J. ; COSTA, W. T. A. ; SALOMAO, R. C. ; CALLEGARI, B. . Intra and Inter-Device Reliabilities of the Instrumented Timed-Up and Go Test Using Smartphones in Young Adult Population. SENSORS, v. 24, p. 2918, 2024.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">6. FERNANDES, T. F. ; CÔRTES, M. I. T. ; PENA, F. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; BELGAMO, A. ; COSTA E SILVA, A. A. ; ANDRÉ DOS SANTOS, CABRAL ; CALLEGARI, B. ; Souza, G. S. . Smartphone-based evaluation of static balance and mobility in type 2 Diabetes. ANAIS DA ACADEMIA BRASILEIRA DE CIÊNCIAS, v. 96, p. 1-1, 2024.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">7. NASCIMENTO, A. Q. ; NAGATA, L. A. R. ; ALMEIDA, M. T. ; COSTA, V. L. S. ; MARIN, A. B. R. ; TAVARES, V. B. ; ISHAK, G. ; CALLEGARI, B. ; SANTOS, E. G. R. ; SOUZA, GIVAGO SILVA ; MELO NETO, J. S. . Smartphone-based inertial measurements during Chester step test as a predictor of length of hospital stay in abdominopelvic cancer postoperative period: a prospective cohort study. World Journal of Surgical Oncology, v. 22, p. 71-1, 2024.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">8. FERREIRA, E. C. V. ; MARQUES, A. P. ; KULIS, S. ; GAJEWSKI, J. ; MORAES, A. A. C. ; DUARTE, M. B. ; ALMEIDA, G. C. S. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; ANDRÉ DOS SANTOS, CABRAL ; Souza, Gilvago Silva ; COSTA E SILVA, A. A ; CALLEGARI, B. . Validity And Reliability of a Mobile Device Application for Assessing Motor Performance in the 30-Second Sit-To-Stand Test. IEEE Access, p. 1-1, 2025.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">9. AZEVEDO, L. S. ; FEITOSA JR, N. Q. ; SANTOS, E. G. R. ; ALVAREZ, M. A. M. ; NORAT, L. A. X. ; BOTELHO, G. I. S. ; BELGAMO, A. ; PINTO, G. H. L. ; SANTANA DE CASTRO, KETLIN JAQUELLINE ; CALLEGARI, B. ; SILVA, A. A. C. E. ; SALOMAO, R. C. ; ANDRÉ DOS SANTOS, CABRAL ; ROSA, A. A. M. ; Silva Souza, Givago . Assessing static balance control improvement following cataract surgery using a smartphone. Digital Health, v. 11, p. 1-1, 2025.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">10. DUARTE, M. B. ; MORAES, A. A. C. ; FERREIRA, E. V. ; ALMEIDA, G. C. S. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; OLIVEIRA, P. R. ; AMORIM, C. F. ; ANDRÉ DOS SANTOS, CABRAL ; SAUNIER, G. J. A. ; COSTA E SILVA, A. A. ; SOUZA, GIVAGO S. ; CALLEGARI, B. . Validity and reliability of a smartphone-based assessment for anticipatory and compensatory postural adjustments during predictable perturbations. GAIT & POSTURE, v. 96, p. 9-17, 2022.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">11. MORAES, A. A. C. ; DUARTE, M. B. ; FERREIRA, E. V. ; ALMEIDA, G. C. S. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; OLIVEIRA, P. R. ; AMORIM, C. F. ; ANDRÉ DOS SANTOS, CABRAL ; COSTA E SILVA, A. A. ; Souza, G. S. ; CALLEGARI, B. . Validity and reliability of smartphone app for evaluating postural adjustments during step initiation. SENSORS, v. 1, p. 1, 2022.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">12. MORAES, A. A. C. ; DUARTE, M. B. ; SANTOS, E. J. M. ; ALMEIDA, G. C. S. ; ANDRÉ DOS SANTOS, CABRAL ; COSTA E SILVA, A. A. ; GARCEZ, D. R. ; GIVAGO DA SILVA, SOUZA ; CALLEGARI, B. . Comparison of inertial records during anticipatory postural adjustments obtained with devices of different masses. PeerJ, v. 11, p. e15627, 2023.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">13. BRITO, F. A. C. ; MONTEIRO, L. C. P. ; SANTOS, E. G. R. ; LIMA, R. C. ; SANTOS-LOBATO, B. L. ; ANDRÉ DOS SANTOS, CABRAL ; CALLEGARI, B. ; SILVA, A. A. C. E. ; GIVAGO DA SILVA, SOUZA . The role of sex and handedness in the performance of the smartphone-based Finger-Tapping Test. PLOS Digital Health, v. 2, p. e0000304, 2023.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">14. LIMA, R. C. ; BRITO, F. A. C. ; NASCIMENTO, R. L. ; MARTINS, S. N. E. S. ; MONTEIRO, L. C. P. ; SEABRA, J. P. ; FARIA, H. L. C. ; SILVA, L. M. C. ; MIRANDA, V. M. S. ; BELGAMO, A. ; ANDRÉ DOS SANTOS, CABRAL ; CALLEGARI, B. ; COSTA E SILVA, A. A ; CRISP, A. ; ALVES, CÂNDIDA HELENA LOPES ; LACERDA, E. M. C. B. ; SOUZA, G.S. . DATASET OF SMARTPHONE-BASED FINGER TAPPING TEST. Scientific Data, v. 1, p. 1, 2024.</a>.</p> <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">15. ALMEIDA, J. R. ; MONTEIRO, L. C. P. ; SOUZA, P. H. C. ; ANDRÉ DOS SANTOS, CABRAL ; BELGAMO, A. ; COSTA E SILVA, A. A ; CRISP, A. ; CALLEGARI, B. ; AVILA, P. E. S. ; SILVA, J. A. ; BASTOS, G. N. T. ; SOUZA, G.S. . Comparison of joint position sense measured by inertial sensors embedded in portable digital devices with different masses. Frontiers in Neuroscience, v. 19, p. 1-1, 2025.</a>.</p> </p> </div> """) st.markdown(html, unsafe_allow_html=True)

# === Referências ===
elif pagina == "📖 Referências bibliográficas":
    html = dedent("""
    <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px;
                margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;">
      <p>Referências do projeto Momentum (como no seu texto original).</p>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)















































