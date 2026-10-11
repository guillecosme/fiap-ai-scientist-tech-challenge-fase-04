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

## Recuperação

| # | Decisão | Motivo e evidência | Onde |
|---|---|---|---|
| D05 | Embeddings `paraphrase-multilingual-MiniLM-L12-v2`, normalizados, sobre o texto limpo | modelo das aulas, com português e 384 dimensões; aproxima paráfrases ("veio quebrado" e "veio com defeito": 0,65). Limitação medida: pesa pouco a negação ("não recebi o produto" fica a 0,61 de "recebi o produto antes do prazo") | notebook 03, `src/indexing/vetorial.py` |
| D06 | FAISS `IndexFlatIP` (busca exata) | 12 ms por busca em 34.889 vetores e 5 ms com filtro; um índice aproximado só trocaria precisão por uma velocidade que não faz falta | notebook 03 |
| D07 | Busca híbrida: BM25 e vetorial combinadas | 63% das avaliações do grupo "não recebeu" não usam nenhuma forma de "receber" ("ainda aguardando", "não chegou"). Na precisão@50 por metadado, o vetorial vence em "não recebi o produto" (69% contra 62% do BM25) e perde em "o pedido chegou atrasado" (56% contra 59%): nenhum ramo domina. A contribuição de cada ramo é medida na ablação | notebooks 02 e 03 |
| D08 | Re-ranking dos 50 candidatos com cross-encoder multilíngue | o bi-encoder traz um elogio ("produto de acordo com anúncio", nota 5) entre os cinco primeiros para "produto diferente do anunciado"; o cross-encoder lê pergunta e texto juntos. O ganho e o custo em latência são medidos na ablação | notebook 03 |
| D09 | Filtros por metadado (nota, situação da entrega, atraso, resposta antes da entrega, categoria, período, UF) antes da busca | 60% dos comentários que falam de entrega têm nota 4 ou 5: sem filtro, uma pergunta sobre reclamações de entrega recupera elogios. A entrega separa satisfeitos de insatisfeitos com efeito grande (nota média de 4,29 no prazo e 2,27 com atraso) | notebooks 01 e 02 |
| D19 | O recorte "não recebeu" inclui os clientes que responderam antes de receber o pedido | 4.770 clientes de pedidos que acabaram entregues responderam antes de receber, com nota média 1,80 e comentários como "ainda não recebi"; o grupo é quase três vezes maior que o de pedidos não entregues (1.661) | notebook 01 |

## Geração

| # | Decisão | Motivo e evidência | Onde |
|---|---|---|---|
| D20 | Respostas de frequência informam o denominador (avaliações com comentário) | quem comenta é mais insatisfeito que a base: 77% de quem dá nota 1 escreve comentário, contra 36% de quem dá nota 5 (V de Cramér = 0,29). Uma proporção calculada sobre o texto não vale para todos os clientes | notebook 01 |
