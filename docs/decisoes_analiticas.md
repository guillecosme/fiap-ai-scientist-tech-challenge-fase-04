# Decisões analíticas

Registro das decisões do projeto, com o motivo, a evidência e o notebook ou módulo onde cada uma foi aplicada. As decisões D01 a D16 foram previstas no plano do projeto e são confirmadas ou ajustadas aqui à medida que a evidência aparece; as decisões a partir de D17 surgiram da análise dos dados.

## Base de conhecimento

| # | Decisão | Motivo e evidência | Onde |
|---|---|---|---|
| D17 | O grão da base é a avaliação (`review_id`); uma avaliação que cobre vários pedidos aparece uma vez, com os pedidos listados | 789 `review_id` estão ligados a mais de um pedido, sempre do mesmo cliente e com nota, texto e datas idênticos. Indexar uma cópia por pedido repetiria a mesma evidência na resposta | notebook 01, `src/data/base.py` |
| D18 | Ausentes não são imputados | comentário ausente é escolha do cliente (58,7% das avaliações); entrega ausente corresponde a pedido não entregue ou cancelado e vira situação de entrega; categoria ausente (2,1%) vira "Sem categoria" | notebook 01 |
| D01 | Um documento por avaliação (título e comentário juntos), sem fragmentação | o documento mediano tem 15 tokens do modelo de embeddings e o percentil 99, 57, contra um limite de 128; só um documento passa do limite, por pontuação repetida | notebook 02, `src/preprocessing/documentos.py` |
| D02 | Avaliações sem texto ficam fora do índice e dentro das estatísticas | 56.040 avaliações são só a nota; outras 277 não têm palavra útil depois da limpeza. Não há o que recuperar, mas elas entram nas proporções de nota e entrega | notebooks 01 e 02 |
| D21 | Títulos sem comentário entram no índice | 1.729 avaliações, quase todas elogios genéricos escritos pelo cliente; a deduplicação junta esses títulos aos grupos de texto igual | notebook 02 |
| D03 | Texto original para exibição; texto limpo para embeddings e contexto do LLM; texto normalizado para deduplicação e índice lexical | a evidência mostrada ao gestor precisa ser literal; o modelo de embeddings não ganha nada com emojis e quebras de linha; o índice lexical precisa de uma forma única para cada palavra | `src/preprocessing/limpeza.py` |
| D04 | Textos idênticos depois de normalizados compartilham uma entrada do índice, com a contagem; o filtro por metadado é aplicado por avaliação | 8.230 avaliações (19,6% das com texto) estão em 1.026 textos repetidos, curtos (mediana de 22 caracteres). 93% desses grupos misturam categorias e 51% misturam notas, por isso o filtro não pode ser aplicado ao grupo | notebook 02, `src/preprocessing/deduplicacao.py` |
| D22 | As negações "não", "nem" e "sem" ficam no índice lexical, apesar de estarem na lista de stopwords do NLTK | "não" é o termo que mais separa notas baixas e altas (61% dos comentários de nota 1 e 2 contra 7% dos de nota 4 e 5); sem ele, "não recebi" e "recebi" seriam a mesma consulta | notebook 02, `src/preprocessing/tokenizacao.py` |

## Mapa de temas

| # | Decisão | Motivo e evidência | Onde |
|---|---|---|---|
| D23 | Dois mapas de temas, por polaridade: textos com nota média até 3 (problemas) e acima de 3 (elogios) | num modelo único, tópicos de entrega juntavam "não foi entregue" e "chegou antes do prazo", porque o embedding pesa pouco a polaridade; só 9,2% dos textos têm nota média estritamente entre 2 e 4, então o corte quase não depende do ponto escolhido | notebook 04, `src/indexing/temas.py` |
| D24 | BERTopic em vez de LDA para o mapa de temas | coerência c_v média de 0,50 e 0,52 (BERTopic) contra 0,46 e 0,44 (LDA), sem significância estatística (p = 0,54 e 0,10); o LDA mistura assuntos em textos de 10 palavras e o BERTopic reaproveita os embeddings da busca. Limitação declarada: a métrica não separa os dois métodos | notebook 04 |
| D10 | Perguntas de frequência respondidas com a contagem por tema dentro do recorte, feita em código | o top-10 recuperado não sustenta "principais problemas" sobre 42 mil textos; o mapa dá a ordem de grandeza (pedido não recebido: 40% das avaliações de problema; defeito e produto errado: 18%; pedido incompleto: 14%) | notebook 04, `theme_frequency` |

## Recuperação

| # | Decisão | Motivo e evidência | Onde |
|---|---|---|---|
| D05 | Embeddings `paraphrase-multilingual-MiniLM-L12-v2`, normalizados, sobre o texto limpo | modelo das aulas, com português e 384 dimensões; aproxima paráfrases ("veio quebrado" e "veio com defeito": 0,65). Limitação medida: pesa pouco a negação ("não recebi o produto" fica a 0,61 de "recebi o produto antes do prazo") | notebook 03, `src/indexing/vetorial.py` |
| D06 | FAISS `IndexFlatIP` (busca exata) | 12 ms por busca em 34.889 vetores e 5 ms com filtro; um índice aproximado só trocaria precisão por uma velocidade que não faz falta | notebook 03 |
| D07 | Busca híbrida: BM25 e vetorial combinadas | 63% das avaliações do grupo "não recebeu" não usam nenhuma forma de "receber" ("ainda aguardando", "não chegou"). Na precisão@50 por metadado, o vetorial vence em "não recebi o produto" (69% contra 62% do BM25) e perde em "o pedido chegou atrasado" (56% contra 59%): nenhum ramo domina. Na ablação, a fusão empata com o vetorial (diferenças não significativas) e supera o BM25 em precisão (+0,05). Limitação declarada: o ganho da fusão em si não foi demonstrado; a híbrida fica porque nenhum ramo domina entre tipos de pergunta | notebooks 02, 03 e 05 |
| D08 | Re-ranking dos 50 candidatos com cross-encoder multilíngue | o bi-encoder traz um elogio ("produto de acordo com anúncio", nota 5) entre os cinco primeiros para "produto diferente do anunciado"; o cross-encoder lê pergunta e texto juntos. Na ablação, precision@10 +0,09 e recall@10 +0,08 (significativos), ao custo de cerca de 0,9 s por pergunta | notebooks 03 e 05 |
| D09 | Filtros por metadado (nota, situação da entrega, atraso, resposta antes da entrega, categoria, período, UF) antes da busca | 60% dos comentários que falam de entrega têm nota 4 ou 5: sem filtro, uma pergunta sobre reclamações de entrega recupera elogios. Na ablação, MRR de 0,65 para 0,86; sem as perguntas de metadado, +0,11 (IC de 0,04 a 0,19); precisão objetiva de 82% a 100% nas perguntas de atraso e não recebimento, contra 45% a 47% sem filtro | notebooks 01, 02 e 05 |
| D25 | HyDE fora do pipeline final | sem ganho significativo em recall e precisão e MRR 0,045 menor, ao custo de uma chamada ao LLM por pergunta | notebook 05 |
| D26 | MMR com λ = 0,7 nos 10 textos que vão ao LLM | similaridade média entre os resultados de 0,53 para 0,49, com precision@10 3,5 pontos menor (parte por textos não julgados) | notebook 05 |
| D19 | O recorte "não recebeu" inclui os clientes que responderam antes de receber o pedido | 4.770 clientes de pedidos que acabaram entregues responderam antes de receber, com nota média 1,80 e comentários como "ainda não recebi"; o grupo é quase três vezes maior que o de pedidos não entregues (1.661) | notebook 01 |

## Geração e orquestração

| # | Decisão | Motivo e evidência | Onde |
|---|---|---|---|
| D11 | Saída estruturada com citação por afirmação, verificada por código | torna a fundamentação conferível sem depender de leitura; validade das citações de 0,97, com 6 afirmações sem evidência retiradas antes de chegar ao gestor | notebook 06, `src/generation/citacoes.py` |
| D12 | Gate de evidência com limiar calibrado (-0,37 no score do cross-encoder) e mínimo de 3 avaliações no recorte | o limiar sai das 35 perguntas de calibração; no teste, 10 de 10 não respondíveis recusadas e 27 de 29 respondíveis respondidas. Limitação: 7 das 60 respondíveis recusadas | notebook 06, `src/graph/gate.py` |
| D13 | Evidência delimitada e declarada como dado, nunca instrução | uma avaliação sintética com uma ordem foi ignorada nas 3 perguntas testadas | `reports/robustez.md` |
| D14 | Grafo LangGraph com arestas condicionais, em vez de agente ReAct | caminho auditável, no máximo 2 chamadas ao LLM por pergunta (1,07 em média nas recusas) e recusa decidida por código | notebook 06, `src/graph/fluxo.py` |
| D15 | Rastro de cada consulta em Parquet | filtros, entradas e scores, decisão, resposta, tempo por etapa e tokens consultáveis depois da execução; base das métricas de operação | `reports/rastros/` |
| D16 | Provedor configurável por variável de ambiente, com modelo de reserva e cache | troca de modelo sem tocar no pipeline; o cache por hash do prompt evita pagar duas vezes pela mesma chamada | `src/generation/provedor.py` |
| D20 | Respostas de frequência informam o denominador (avaliações com comentário) | quem comenta é mais insatisfeito que a base: 77% de quem dá nota 1 escreve comentário, contra 36% de quem dá nota 5 (V de Cramér = 0,29). Uma proporção calculada sobre o texto não vale para todos os clientes | notebook 01 |

## O que ficou fora por decisão

| Técnica | Motivo |
|---|---|
| Chunking | os comentários cabem inteiros no modelo de embeddings (D01) |
| Índice aproximado (HNSW, IVF) | a busca exata leva 12 ms nesta escala (D06) |
| HyDE no pipeline final | sem ganho na ablação (D25) |
| Fine-tuning de modelo | o RAG responde com a base sem treinar modelo; custo e prazo não se justificam |
| Agente ReAct livre | não determinístico e sem ganho para busca numa base única (D14) |
| Classificador supervisionado de sentimento (Naive Bayes, regressão logística) | a nota do cliente já é o rótulo de satisfação |
| Multimodal | a base não tem imagens |
| Interface gráfica | a API com documentação interativa atende à demonstração |
