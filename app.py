"""Interface Streamlit: perguntas com evidencias, qualidade medida e auditoria do juiz.

Uso:
    make app        # abre em http://localhost:8501

Usa o mesmo pipeline da API (`src/graph/fluxo.py`). Sem as variaveis do LLM no ambiente
ou no .env, a analise da pergunta cai para as regras e a geracao fica indisponivel.
"""

import os

import streamlit as st

from src.app import servicos

os.environ.setdefault("TQDM_DISABLE", "1")

st.set_page_config(page_title="Voice of Customer Intelligence", layout="wide")


@st.cache_resource(show_spinner="Carregando índices e modelos...")
def pipeline():
    p, aviso = servicos.build_pipeline()
    p.retriever.retrieve("entrega", k=1)  # carrega os modelos antes da primeira pergunta
    return p, aviso


st.title("Voice of Customer Intelligence")
st.caption("Avaliações de clientes da Olist (pedidos de 2016 a 2018) respondendo a perguntas "
           "de gestores, com as evidências de cada afirmação.")

aba_pergunta, aba_qualidade, aba_auditoria = st.tabs(
    ["Perguntar", "Qualidade medida", "Auditoria do juiz"])

# --------------------------------------------------------------------------- perguntar
with aba_pergunta:
    p, aviso = pipeline()
    if aviso:
        st.info(f"Modelo de linguagem indisponível ({aviso}). A busca funciona; a geração não.")

    with st.sidebar:
        st.header("Como a resposta é construída")
        st.markdown(
            "1. **Análise:** o LLM identifica escopo, recorte (nota, entrega, categoria, "
            "período, região) e reescreve a pergunta como um cliente escreveria.\n"
            "2. **Recuperação:** filtros, BM25 e busca vetorial fundidos por RRF, "
            "re-ranking com cross-encoder e MMR.\n"
            "3. **Gate de evidência:** recusa recorte vazio, recorte pequeno ou evidência "
            f"fraca (limiar {servicos.br(p.gate.threshold)} no cross-encoder).\n"
            "4. **Geração:** cada afirmação cita as avaliações que a sustentam.\n"
            "5. **Verificação:** citações conferidas por código.")
        if p.llm is not None:
            st.caption(f"Modelo: {p.llm.model}")

    grupo = st.selectbox("Exemplos", ["Escrever minha pergunta", *servicos.EXEMPLOS])
    if grupo == "Escrever minha pergunta":
        pergunta = st.text_input("Pergunta", placeholder="Ex.: o que os clientes dizem sobre "
                                                         "a embalagem dos produtos?")
    else:
        pergunta = st.radio("Pergunta", servicos.EXEMPLOS[grupo], label_visibility="collapsed")

    if st.button("Perguntar", type="primary", disabled=not pergunta):
        with st.spinner("Analisando a pergunta, buscando e gerando a resposta..."):
            st.session_state["estado"] = p.run(pergunta.strip())

    s = st.session_state.get("estado")
    if s:
        titulo, tipo = servicos.decision(s)
        getattr(st, tipo)(titulo)
        resposta = s.get("resposta") or {}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recorte", s.get("recorte") or "não se aplica")
        total = s.get("total_recorte")
        c2.metric("Avaliações no recorte", f"{total:,}".replace(",", ".") if total else "-")
        c3.metric("Tempo", f"{servicos.br(s['tempos']['total'], 1)} s")
        c4.metric("Chamadas ao LLM", (s.get("llm") or {}).get("chamadas", 0))

        st.subheader("Resposta")
        st.markdown(resposta.get("resumo", ""))
        for a in resposta.get("afirmacoes", []):
            st.markdown(f"- {a['texto']}  `{' '.join(a['evidencias'])}`")
        if resposta.get("limitacoes") and s.get("decisao") == "responder":
            st.caption(f"Limitações: {resposta['limitacoes']}")

        freq = servicos.frequency_table(s)
        if not freq.empty:
            st.subheader("Frequência por tema no recorte")
            st.caption("Contada em código sobre todas as avaliações do recorte, não só "
                       "sobre as recuperadas.")
            st.bar_chart(freq.set_index("Tema")["Avaliações"], horizontal=True,
                         color="#ED145B")

        evid = servicos.cited_evidence(s)
        if evid:
            st.subheader("Evidências citadas")
            for e in evid:
                with st.container(border=True):
                    nota = servicos.score_label(e["nota"])
                    st.markdown(f"**{e['id']}** · {nota} · {e['data']} · "
                                f"{e['categoria']} · {e['entrega']}")
                    st.markdown(f"> {e['texto']}")
                    extra = (f" · {e['n_avaliacoes']} avaliações com o mesmo texto"
                             if e["n_avaliacoes"] > 1 else "")
                    st.caption(f"review_id {e['review_ids'][0]}{extra}")

        with st.expander("Rastro da consulta"):
            analise = s.get("analise") or {}
            st.markdown(f"**Consulta reescrita:** {analise.get('consulta', '')}")
            st.markdown(f"**Análise feita por:** {s.get('fonte_analise', '')}")
            st.json(s.get("filtros") or {})
            st.markdown("**Tempo por etapa (s)**")
            st.json(servicos.stage_times(s))
            tabela = servicos.retrieval_table(s)
            if not tabela.empty:
                st.markdown("**Entradas recuperadas**")
                st.dataframe(tabela, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- qualidade
with aba_qualidade:
    q = servicos.quality_summary()
    st.markdown("Resultados do conjunto de avaliação de 80 perguntas (60 respondíveis e 20 sem "
                "resposta possível), definido antes da construção. Detalhes nos notebooks 05 "
                "e 06.")
    abl = q.get("ablacao")
    final = q.get("final")
    linha1 = st.columns(2)
    linha2 = st.columns(2)
    if final and abl is not None:
        bm25 = abl.set_index("variante").loc["bm25", "mrr"]
        linha1[0].metric("MRR do pipeline final", servicos.br(final["mrr"]),
                         delta=f"{servicos.br(final['mrr'] - bm25)} sobre o BM25")
    if "fidelidade" in q:
        linha1[1].metric("Afirmações sustentadas", f"{q['fidelidade']:.0%}")
    if "validade" in q:
        linha2[0].metric("Citações válidas", f"{q['validade']:.0%}")
    if "recusa" in q:
        linha2[1].metric("Sem resposta recusadas (teste)", f"{q['recusa'][0]} de {q['recusa'][1]}",
                         help=f"Respondíveis respondidas no teste: {q['respondidas'][0]} de "
                              f"{q['respondidas'][1]}")
    if abl is not None:
        st.subheader("Ablação da recuperação")
        st.dataframe(servicos.ablation_view(abl), hide_index=True, width="stretch")
        st.image("images/05_ablacao_recuperacao.png")
    st.subheader("Gate de evidência")
    st.image("images/06_gate_de_evidencia.png")
    if "tempo_mediano" in q:
        st.caption(f"Tempo mediano por resposta: {servicos.br(q['tempo_mediano'], 1)} s "
                   f"(análise, recuperação e geração).")

# --------------------------------------------------------------------------- auditoria
with aba_auditoria:
    st.markdown(
        "Amostra de 120 pares pergunta e avaliação para conferir o juiz LLM. A nota do juiz "
        "fica escondida até você rotular. Rubrica: **2** a avaliação serviria de evidência "
        "para a pergunta; **1** toca no assunto de forma vaga; **0** não trata do assunto. "
        "Ignore restrições de nota, categoria, período e região.")
    try:
        df = servicos.load_audit()
    except FileNotFoundError:
        st.warning("Arquivo de auditoria não encontrado.")
        st.stop()
    feitos = int((df["nota_manual"] != "").sum())
    st.progress(feitos / len(df), text=f"{feitos} de {len(df)} pares rotulados")
    i = servicos.next_unlabeled(df)
    if i is not None:
        linha = df.loc[i]
        with st.container(border=True):
            st.markdown(f"**Pergunta:** {linha['pergunta']}")
            st.markdown(f"> {linha['texto']}")
        c0, c1, c2 = st.columns(3)
        for col, nota, rotulo in [(c0, 0, "0 · irrelevante"), (c1, 1, "1 · parcial"),
                                  (c2, 2, "2 · relevante")]:
            if col.button(rotulo, key=f"nota{nota}", width="stretch"):
                servicos.save_label(df, i, nota)
                st.rerun()
    else:
        st.success("Todos os pares foram rotulados.")
    acordo = servicos.audit_agreement(df)
    if acordo["rotulados"]:
        st.subheader("Concordância com a sua anotação")
        c1, c2 = st.columns(2)
        c1.metric("Juiz principal", f"{acordo['concordancia_juiz']:.0%}",
                  help=f"kappa de Cohen: {acordo['kappa_juiz']:.2f}")
        c2.metric("Segundo juiz", f"{acordo['concordancia_segundo_juiz']:.0%}",
                  help=f"kappa de Cohen: {acordo['kappa_segundo_juiz']:.2f}")
        with st.expander("Pares rotulados"):
            st.dataframe(df[df["nota_manual"] != ""][
                ["question_id", "texto", "nota_manual", "nota_juiz", "nota_segundo_juiz"]],
                hide_index=True, width="stretch")
