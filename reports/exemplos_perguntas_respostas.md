# Exemplos de perguntas e respostas

Gerado pelo notebook 07 com o pipeline final (análise, recuperação híbrida com filtros e re-ranking, gate calibrado, geração com citações verificadas). As evidências são avaliações reais da base da Olist; os textos aparecem limpos de quebras de linha e emojis.


### Quais são os principais problemas relatados pelos clientes?

**Decisão:** responder
**Recorte:** nota de 1 a 3 · 14.431 avaliações com texto

**Resposta:** Os principais problemas relatados nas avaliações de nota 1 a 3 concentram-se no não recebimento do pedido, defeitos ou recebimento de produtos errados/de má qualidade, e itens faltantes.

| Tema | Avaliações | % |
|---|---|---|
| Pedido não recebido | 5.551 | 40% |
| Defeito, produto errado ou má qualidade | 2.509 | 18% |
| Pedido incompleto, faltando itens | 2.004 | 14% |
| Cor, tamanho ou aparência diferente do anúncio | 955 | 7% |
| Atraso e prazo de entrega longo | 950 | 7% |
| Satisfeito, com ressalvas | 780 | 6% |

- No universo de 14.431 avaliações com texto do recorte (notas de 1 a 3), o problema mais frequente é 'Pedido não recebido', representando 40% dos casos (5.551 avaliações). [E5]
- A categoria de 'Defeito, produto errado ou má qualidade' é o segundo maior problema, correspondendo a 18% das avaliações (2.509 avaliações). [E1, E4, E6, E7, E10]
- Problemas de atraso na entrega e falta de comunicação/satisfação sobre o atraso também são relatados pelos clientes. [E2, E3, E8, E9]

*Limitações:* As porcentagens de frequência foram calculadas especificamente sobre a base de 14.431 avaliações que contêm texto e possuem nota entre 1 e 3, não representando a totalidade de vendas ou avaliações da plataforma.

| Evidência | Avaliação do cliente | Nota | Data | Categoria | Entrega | Iguais |
|---|---|---|---|---|---|---|
| E1 | Mandou o produto errado e ate hoje não entrou em contado para tentar solucionar o problema, Péssima loja. | 1 | 2017-08-30 | Esporte e lazer | Entregue no prazo | 1 |
| E2 | Atrasado. Receber com atraso é sempre ruim. O produto deveria ser entregue até 26/04/2018 e até o momento, não foi. | 2 | 2018-04-29 | Casa e construção | Entregue com atraso | 1 |
| E3 | atrasou más recebi | 3 | 2017-05-30 | Ferramentas e jardim | Entregue no prazo | 1 |
| E4 | Produto errado. Parceiro ruim e metodologia da lannister péssimo. Já tem quase um mês que recebi o produto errado, tive de comprar novamente e as Lojas lannister não tomam nenhuma atitude, só ficam enviando pesquis | 1 | 2018-06-03 | Ferramentas de construção | Entregue no prazo | 1 |
| E5 | Ainda não recebi esta atrasado nunca aconteceu isso sempre como ,chega antes ou no prazo. | 3 | 2017-10-29 | Telefonia | Não entregue | 1 |
| E6 | Acabamento ruim e estampa errada. | 1 | 2017-06-06 | Calçados | Entregue no prazo | 1 |
| E7 | Não gostei tecido ruim péssimo | 1 | 2017-12-14 | Cama, mesa e banho | Entregue no prazo | 1 |
| E8 | péssimo *. mercadoria super atrasada e nem satisfação | 2 | 2018-06-09 | Cama, mesa e banho | Entregue com atraso | 1 |
| E9 | Não recebi nem uma satisfação pelo atraso | 1 | 2017-12-19 | Utilidades domésticas | Entregue com atraso | 1 |
| E10 | Qualidade muito ruim. Não encaixa.pessimo | 2 | 2018-03-14 | Automotivo | Entregue no prazo | 1 |

*review_id das evidências:* E1: 9dd82c651799bbaeabfd11deed989cd4; E2: e057360360ac06fb9abc25efe4505413; E3: 03a9a298cbe3fe2c3adf6d776c8d8469; E4: 4a93de6c66dee67442f7cbd44428f399; E5: 6e25aed0cc41aa34f87e0842b28c63dd; E6: 0a18dc9b77e58490b51d75a5b97ad30d; E7: 56c6c12a05e68f6821ccc9d8f264d3f3; E8: 7a93610296d0a660555dbb57d6b9024c; E9: 7015ae1a43e51e25b13198625e8ccd91; E10: 629a83b85a131bc017dd8c31a2b54043
*Tempo:* 1.5 s nesta execução · chamadas ao LLM: 2

---

### Quais reclamações relacionadas à entrega aparecem com maior frequência?

**Decisão:** responder
**Recorte:** nota de 1 a 3 · 14.431 avaliações com texto

**Resposta:** Entre as reclamações de entrega no recorte de notas de 1 a 3, a não entrega do pedido é o problema mais frequente, seguida pelo atraso e prazos longos.

| Tema | Avaliações | % |
|---|---|---|
| Pedido não recebido | 5.551 | 40% |
| Defeito, produto errado ou má qualidade | 2.509 | 18% |
| Pedido incompleto, faltando itens | 2.004 | 14% |
| Cor, tamanho ou aparência diferente do anúncio | 955 | 7% |
| Atraso e prazo de entrega longo | 950 | 7% |
| Satisfeito, com ressalvas | 780 | 6% |

- A reclamação sobre entrega com maior frequência no recorte de 14.431 avaliações (notas 1 a 3) é 'Pedido não recebido', representando 40% dos casos (5.551 avaliações). Os clientes relatam que o produto não chegou ou que não o receberam. [E1, E2, E3, E6, E7]
- A segunda principal reclamação ligada à entrega é 'Atraso e prazo de entrega longo', correspondendo a 7% do total do recorte (950 avaliações). Os clientes mencionam a demora na entrega e a necessidade de buscar os produtos nos Correios. [E4, E6, E7, E8, E10]
- Diversos relatos atribuem os problemas de entrega e atraso especificamente aos Correios, sugerindo o uso de transportadoras privadas. [E1, E2, E3, E5, E7, E8, E9]

*Limitações:* As contagens e porcentagens referem-se estritamente ao conjunto de 14.431 avaliações com texto do recorte de notas de 1 a 3, não representando a totalidade de vendas ou avaliações da plataforma.

| Evidência | Avaliação do cliente | Nota | Data | Categoria | Entrega | Iguais |
|---|---|---|---|---|---|---|
| E1 | nao recebi o produto acredito ser problema dos correios prazo de entrega expirou | 1 | 2017-07-22 | Marketplace | Entregue com atraso | 1 |
| E2 | Problemas na entrega. Não seria justo não recomendar a loja pois não recebi o produto pois o Correios não entregou. Talvez seria interessante a loja utilizar outra transportadora. | 1 | 2018-05-09 | Informática e acessórios | Não entregue | 1 |
| E3 | NAO CHEGOU POR PROBLEMAS NOS CORREIOS | 1 | 2018-03-15 | Telefonia | Entregue no prazo | 1 |
| E4 | Nao gostei. Por motivo da demora na entrega e tive que busca no correio | 1 | 2018-01-17 | Consoles e games | Entregue no prazo | 1 |
| E5 | A empresa deveria usar uma transportadora, pois os correios não tem competência pra envio além da demora corre o risco de não entregar. | 1 | 2017-06-02 | Automotivo | Entregue no prazo | 1 |
| E6 | atraso na entrega. não recebi a entrega ainda. | 1 | 2018-05-19 | Beleza e saúde | Entregue no prazo | 1 |
| E7 | Atraso no entrega. Acredito que tenha sido atraso dos correios , ainda não recebi o produto . | 2 | 2018-05-05 | Esporte e lazer | Não entregue | 1 |
| E8 | Muito ruim a demora na entrega. Se fosse por transportadora seria mesmo valor e muito mais rápido a entrega. | 2 | 2018-02-16 | Utilidades domésticas | Entregue no prazo | 1 |
| E9 | O correio informou problemas com a entrega.encaminhei 2 mensagens e não tive retorno.Recebi uma ligação no sábado das lannister que a entrega seria no prazo.mas até o momento nada. | 1 | 2018-03-07 | Cama, mesa e banho | Entregue com atraso | 1 |
| E10 | ATRASO ENORME NA ENTREGA. A entrega está muuuuito atrasada e não recebi nenhum retorno a respeito. | 1 | 2018-07-27 | Utilidades domésticas | Entregue com atraso | 1 |

*review_id das evidências:* E1: 05481f963c19afbeeb6c2d48fb31f3fe; E2: b7c9e20cf48170ef505e502cae85a46d; E3: 9183ee1d6da9be1da5e0776fc4b023d6; E4: 5cb48a77d5334f597153e0de0648bb6e; E5: de954d5e671955bf0500f4328055b876; E6: 93623428a858a3a0cf8679c420c4a8d3; E7: ac6bdc8a49173f8db773a6a97fc6d087; E8: c40ac9052280bd0ef25f33876f68797d; E9: 22162f5d35b60c13b619f8ed0458adfd; E10: f67c215f6e2b0e5d7d4cfdd598e823ae
*Tempo:* 1.6 s nesta execução · chamadas ao LLM: 2

---

### O que os clientes mais elogiam em suas experiências de compra?

**Decisão:** responder
**Recorte:** nota de 4 a 5 · 27.662 avaliações com texto

**Resposta:** Os clientes elogiam principalmente o recebimento do produto e a entrega conforme o combinado, a pontualidade/rapidez no envio e a boa qualidade dos produtos adquiridos.

| Tema | Avaliações | % |
|---|---|---|
| Produto e entrega conforme o combinado | 7.429 | 27% |
| Entrega antes ou dentro do prazo | 7.221 | 26% |
| Elogio curto e genérico | 6.061 | 22% |
| Satisfeito, com ressalvas | 2.468 | 9% |
| Entrega rápida | 2.159 | 8% |
| Atendimento, site e confiança na loja | 866 | 3% |

- No conjunto de 27.662 avaliações de nota 4 a 5, os temas mais recorrentes são 'Produto e entrega conforme o combinado' (7.429 avaliações, ou 27%) e 'Entrega antes ou dentro do prazo' (7.221 avaliações, ou 26%). [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10]
- Nas evidências textuais da amostra, os consumidores destacam de forma recorrente a agilidade da entrega somada à ótima qualidade dos produtos. [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10]
- O atendimento de excelência também é ressaltado por clientes em alguns dos relatos com nota máxima. [E3, E4]

*Limitações:* As porcentagens apresentadas referem-se estritamente ao universo de 27.662 avaliações com texto contidas no recorte de nota 4 a 5. As evidências textuais individuais (E1 a E10) representam apenas uma amostra desse universo.

| Evidência | Avaliação do cliente | Nota | Data | Categoria | Entrega | Iguais |
|---|---|---|---|---|---|---|
| E1 | Excelente compra produto de qualidade e entrega rápida recomendo. | 5 | 2017-10-01 | Utilidades domésticas | Entregue no prazo | 1 |
| E2 | relógio exercito. excelente produto,ótima qualidade e rapidez na entrega ! recomendo com toda certeza ! | 5 | 2018-06-12 | Relógios e presentes | Entregue no prazo | 1 |
| E3 | Super recomendo. Entrega rápida, bom atendimento e produto de qualidade | 5 | 2018-08-14 | Instrumentos musicais | Entregue no prazo | 1 |
| E4 | Atendimento excelente, extremamente rápido, produto de ótima qualidade, superou minhas expectativas em todos os aspectos. | 5 | 2018-03-01 | Sem categoria | Entregue no prazo | 1 |
| E5 | entrega rápida e produto de boa qualidade, recomendo | 5 | 2018-03-10 | Ferramentas e jardim | Entregue no prazo | 1 |
| E6 | Super recomendo, entrega rápida. Produto ótimo. | 5 | 2017-05-04 | Cool stuff | Entregue no prazo | 1 |
| E7 | Ótimo produto excelente qualidade entrega rápida. | 5 | 2018-04-10 | Bebês | Entregue no prazo | 1 |
| E8 | Ótima qualidade entrega muito rápida recomendo | 5 | 2017-04-13 | Cinema e foto | Entregue no prazo | 1 |
| E9 | Produto ótimo, entrega rapida, super recomendo | 5 | 2017-07-18 | Beleza e saúde | Entregue no prazo | 1 |
| E10 | super recomendo. excelente produto, entrega rápida. recomendo. | 5 | 2018-07-14 | Casa e construção | Entregue no prazo | 1 |

*review_id das evidências:* E1: b0f9a68177c8390ca39707a56e998772; E2: 752d6729f1d725fffda68151f3756b6b; E3: 78c58d4a6cc2e756b43e6ea1ff6c6229; E4: cea5891cf0a15fbe6c837f350f697259; E5: 31bdc52857f8f7dcfd20484ee64c6e97; E6: a26ba95a5def3814a0389cbd23e2fb5a; E7: 08b8ab4cd13b5ce55a4f6558ba1d1c87; E8: 51359b855458a464aaf21671ff41fd0f; E9: 7046fbb96a01a1a4352ed2dac74f52cf; E10: 8aa5905667b126c8cd113a1ab88ecf74
*Tempo:* 1.1 s nesta execução · chamadas ao LLM: 2

---

### Quais problemas são relatados por clientes insatisfeitos?

**Decisão:** responder
**Recorte:** nota de 1 a 2 · 10.837 avaliações com texto

**Resposta:** Os clientes insatisfeitos (notas 1 a 2) relatam principalmente a não entrega de pedidos, defeitos ou recebimento de produtos errados, além de atrasos e suporte deficiente.

| Tema | Avaliações | % |
|---|---|---|
| Pedido não recebido | 4.797 | 44% |
| Defeito, produto errado ou má qualidade | 1.915 | 18% |
| Pedido incompleto, faltando itens | 1.588 | 15% |
| Atraso e prazo de entrega longo | 653 | 6% |
| Cor, tamanho ou aparência diferente do anúncio | 651 | 6% |
| Acessórios eletrônicos que não funcionam | 434 | 4% |

- O não recebimento do pedido é a principal reclamação, representando 44% das 10.837 avaliações do recorte de notas 1 a 2, também evidenciado em casos específicos. [E2, E6, E10]
- Defeitos no produto, itens de má qualidade, quebrados ou faltando peças correspondem a 18% do total do recorte e são relatados explicitamente por clientes. [E3, E5, E8, E9]
- Envio de produto errado e falhas no atendimento ou suporte das lojas também causam forte insatisfação. [E1, E3, E4, E7]
- Atrasos na entrega representam 6% das avaliações do recorte e são citados como motivo de queixa. [E4, E5, E6, E10]

*Limitações:* As porcentagens apresentadas referem-se estritamente ao conjunto de 10.837 avaliações com texto contendo notas de 1 a 2. As evidências qualitativas recuperadas representam uma amostragem desse recorte.

| Evidência | Avaliação do cliente | Nota | Data | Categoria | Entrega | Iguais |
|---|---|---|---|---|---|---|
| E1 | Mandou o produto errado e ate hoje não entrou em contado para tentar solucionar o problema, Péssima loja. | 1 | 2017-08-30 | Esporte e lazer | Entregue no prazo | 1 |
| E2 | nao recebi produto. pessimo | 1 | 2018-05-01 | Utilidades domésticas | Entregue no prazo | 1 |
| E3 | Zero. O produto chegou com defeito é o site da lannister é péssimo pra trocas, suporte péssimo ! | 1 | 2018-07-06 | Relógios e presentes | Entregue no prazo | 1 |
| E4 | péssimo, além da demora ainda mandaram o produto errado. | 1 | 2017-07-22 | Cama, mesa e banho | Entregue no prazo | 1 |
| E5 | Atrasado e com defeito. Produto chegou atrasado, mal embalado e faltando um Ilhós. | 1 | 2018-07-21 | Bebês | Entregue no prazo | 1 |
| E6 | meu produto nao chegou esta atrasado. | 1 | 2018-06-23 | Beleza e saúde | Entregue com atraso | 1 |
| E7 | Recebi o produto errado! Recebi o produto errado e a central de atendimento é muito ruim, não consigo entrar em contato, respondem email quando querem e não resolvem meu problema! | 1 | 2018-07-06 | Informática e acessórios | Entregue no prazo | 1 |
| E8 | Não recomendo esse produto. Ele é simplesmente péssimo ou veio com defeito. É muito frágil e o som é muito baixo. Não gostei. | 1 | 2018-03-15 | Cool stuff | Entregue no prazo | 1 |
| E9 | Defeito produto. Recebi o pingente com defeito, quebrado. | 1 | 2018-08-25 | Bolsas e acessórios | Entregue no prazo | 1 |
| E10 | Atraso. Ainda não recebi o produto. | 1 | 2018-06-03 | Beleza e saúde | Entregue com atraso | 1 |

*review_id das evidências:* E1: 9dd82c651799bbaeabfd11deed989cd4; E2: 0f80e3076617983291eb9fd9c1dc3498; E3: 2e1d71fce98b42bf6cbdd5e06798c286; E4: 5cebd0cb812691be648e2be73527bfce; E5: 6298846cdf0eceb7ba5661c3810725b7; E6: 4a7118b1f9c3d06869e0cd9d5c1b4ccb; E7: ac313070ee4ecde0406fec694d8bc9cb; E8: 2f71ad6404923ebd6cc31b8654196ef6; E9: 5b14c151dac1c3e27a55c5c512a684b0; E10: 4cc9dc519327a019f0193e75e34c851d
*Tempo:* 2.1 s nesta execução · chamadas ao LLM: 2

---

### Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?

**Decisão:** responder
**Recorte:** sem filtro · 42.093 avaliações com texto

**Resposta:** Sim, a qualidade dos produtos é citada negativamente em 6% (2.517) das 42.093 avaliações com texto, apresentando padrões como chegada de itens danificados/quebrados, divergência do anúncio e material fraco.

| Tema | Avaliações | % |
|---|---|---|
| Produto e entrega conforme o combinado | 7.500 | 18% |
| Entrega antes ou dentro do prazo | 7.379 | 18% |
| Elogio curto e genérico | 6.235 | 15% |
| Pedido não recebido | 5.566 | 13% |
| Defeito, produto errado ou má qualidade | 2.517 | 6% |
| Satisfeito, com ressalvas | 2.473 | 6% |

- O tema 'Defeito, produto errado ou má qualidade' representa 6% (2.517) do total de 42.093 avaliações com texto no recorte geral. [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10]
- Um padrão recorrente é a entrega de produtos que já chegam quebrados, com defeito ou com embalagem inadequada. [E1, E4, E5, E6, E7, E9]
- Outro padrão identificado é o recebimento de produtos com materiais considerados fracos, de baixa qualidade ou incompatíveis com o que foi comprado. [E2, E3, E5, E7, E9, E10]

*Limitações:* As evidências qualitativas recuperadas focam essencialmente nas reclamações de baixa qualidade, defeitos e avarias, não detalhando os aspectos positivos dos 18% de clientes que relataram receber o produto conforme o combinado.

| Evidência | Avaliação do cliente | Nota | Data | Categoria | Entrega | Iguais |
|---|---|---|---|---|---|---|
| E1 | maleta do produto de péssima qualidade, e veio quebrada | 1 | 2017-08-17 | Ferramentas e jardim | Entregue no prazo | 1 |
| E2 | produto de péssima qualidade a estampa do skate veio errada capacete fraco, proteções fracas | 1 | 2018-01-04 | Brinquedos | Entregue com atraso | 1 |
| E3 | Produto de péssima qualidade material muito ruin | 1 | 2018-01-16 | Cama, mesa e banho | Entregue no prazo | 1 |
| E4 | Produto de péssima qualidade, ja chegou com o cronômetro quebrado e os botões com defeito | 1 | 2017-12-14 | Relógios e presentes | Entregue no prazo | 1 |
| E5 | Produto entregue não condiz com o produto comprado, péssima qualidade, assessório veio quebrado. | 1 | 2017-09-20 | Cama, mesa e banho | Entregue no prazo | 1 |
| E6 | Produto com defeito. Produto veio quebrado | 3 | 2018-07-31 | Automotivo | Entregue no prazo | 1 |
| E7 | Recebi apenas um produto de dois que comprei e este veio com vários defeitos demonstrando ser um produto de baixa qualidade | 1 | 2017-10-21 | Brinquedos | Entregue no prazo | 1 |
| E8 | Péssima muito Ruin | 1 | 2018-07-05 | Esporte e lazer | Não entregue | 1 |
| E9 | Produto mal embalado. Produto chegou quebrado é material não é bom | 3 | 2018-05-15 | Móveis e decoração | Entregue no prazo | 1 |
| E10 | Produto errado,cor e tamanho errado, péssima qualidade. | 1 | 2017-12-01 | Cama, mesa e banho | Entregue no prazo | 1 |

*review_id das evidências:* E1: 7c2bfdc27c016b8ce15f3b1e37d830a2; E2: 0c6521a87c0f07246bbc4875a96c3da1; E3: c8bb07efb7fa82eba86e029c80be5d9a; E4: 0be07054d23f0f335ee4d15f4ee9886f; E5: fb9d26ae601018d2c1253e9af35b14e7; E6: 13dcde46c1a250aaed407dd6d5a53b77; E7: 3260e0b3b6cf9b9e8956ff89cc08a189; E8: 4221f3f3b5ffcf57d8eb3497d3332479; E9: c234fb41ba6510ff090beeb2462c0499; E10: 2f2a5e457479ae6caa49462daad14e5d
*Tempo:* 1.4 s nesta execução · chamadas ao LLM: 2

---

### O que os clientes relatam sobre problemas relacionados à entrega?

**Decisão:** responder
**Recorte:** nota de 1 a 3 · 14.431 avaliações com texto

**Resposta:** Os clientes com avaliações insatisfeitas (notas 1 a 3) reclamam predominantemente do atraso na entrega e do fato de não terem recebido os produtos dentro do prazo estipulado.

- Clientes relatam com frequência o atraso na entrega e o não recebimento do produto após o vencimento do prazo estipulado. [E1, E2, E3, E4, E5, E6, E7, E9, E10]
- Além do atraso na entrega, há relatos de falta de posição, comunicação ou satisfação sobre o status do pedido não entregue. [E1, E8]
- Há relato atribuindo o problema de atraso na entrega ao serviço dos Correios. [E3]

*Limitações:* As evidências analisadas representam uma amostra de avaliações com notas de 1 a 3 e não cobrem a totalidade da base de pedidos da empresa.

| Evidência | Avaliação do cliente | Nota | Data | Categoria | Entrega | Iguais |
|---|---|---|---|---|---|---|
| E1 | atraso na entrega. nao recebi meu produto e nem uma posição sobre o mesmo | 1 | 2018-06-03 | Iluminação | Cancelado ou indisponível | 1 |
| E2 | Atraso na entrega, até o momento nao recebi o meu produto. | 1 | 2018-04-05 | Pet shop | Entregue com atraso | 1 |
| E3 | nao recebi o produto acredito ser problema dos correios prazo de entrega expirou | 1 | 2017-07-22 | Marketplace | Entregue com atraso | 1 |
| E4 | Atraso na entrega. Ainda não recebi o produto. | 2 | 2018-07-18 | Relógios e presentes | Não entregue | 1 |
| E5 | entrega atrasada. o prazo terminou e nao recebi meu produto | 1 | 2018-06-01 | Informática e acessórios | Entregue com atraso | 1 |
| E6 | Atraso entrega. Não recebi o produto | 1 | 2018-06-06 | Automotivo | Entregue com atraso | 1 |
| E7 | Ainda nao recebi o produto e ja se encontra atrasado | 1 | 2017-04-21 | Perfumaria | Entregue no prazo | 1 |
| E8 | Muito atraso na entrega de um produto simples e pequeno. Até hoje não recebi nem o produto, nem uma satisfação sobre o assunto. | 1 | 2017-03-22 | Sem categoria | Cancelado ou indisponível | 1 |
| E9 | Atraso na entrega. Ainda não recebi o produto é já se passou o prazo solicitado. | 1 | 2018-08-18 | Esporte e lazer | Entregue com atraso | 1 |
| E10 | demora na entrega, ainda não recebi o produto | 2 | 2018-04-14 | Telefonia | Não entregue | 1 |

*review_id das evidências:* E1: b5e71e87bfbbcd2a826b9610bac99680; E2: 98eaf3a20ac997b34bf433ac0979f4e2; E3: 05481f963c19afbeeb6c2d48fb31f3fe; E4: 1f3f5e3000ed6307fbda490057bfa59b; E5: 1bfec3e913196c72c1ea06fc8816bd9c; E6: 9d6c1f490b2b4bfef33820b5679e2d82; E7: e99c29f0a950abd77239b15adaaa38b8; E8: c99bd47a251c0fef5b5933b293acea03; E9: 0162340d190549d567ee0fdfb8fb725b; E10: d5ec0edbe1505d4f8c80f52d9c19f98d
*Tempo:* 1.9 s nesta execução · chamadas ao LLM: 2

---

### Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?

**Decisão:** responder
**Recorte:** nota de 1 a 3 · 14.431 avaliações com texto

**Resposta:** As avaliações mostram reclamações tanto de prazos de entrega considerados excessivamente longos no momento da compra quanto de atrasos efetivos na entrega do produto.

| Tema | Avaliações | % |
|---|---|---|
| Pedido não recebido | 5.551 | 40% |
| Defeito, produto errado ou má qualidade | 2.509 | 18% |
| Pedido incompleto, faltando itens | 2.004 | 14% |
| Cor, tamanho ou aparência diferente do anúncio | 955 | 7% |
| Atraso e prazo de entrega longo | 950 | 7% |
| Satisfeito, com ressalvas | 780 | 6% |

- Na distribuição do recorte de notas de 1 a 3 (14.431 avaliações com texto), o tema 'Atraso e prazo de entrega longo' representa 7% (950 avaliações), enquanto 'Pedido não recebido' representa 40% (5.551 avaliações). [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10]
- Clientes reclamam de atrasos reais no cumprimento da entrega ou que a entrega não chegou na data prevista. [E1, E7, E8, E9, E10]
- Há insatisfação com a demora da entrega e com o estipulado prazo de entrega ser considerado longo ou absurdo, mesmo quando o produto é entregue dentro do prazo estimado. [E2, E3, E4, E5, E6]

*Limitações:* As evidências apresentadas são uma amostra qualitativa extraída do conjunto total de avaliações com notas de 1 a 3.

| Evidência | Avaliação do cliente | Nota | Data | Categoria | Entrega | Iguais |
|---|---|---|---|---|---|---|
| E1 | atraso na entrega da mercadoria,não cumpre prazo determinado. | 1 | 2018-08-24 | Cama, mesa e banho | Entregue com atraso | 1 |
| E2 | O prazo de entrega é muito longo, demora demais. | 2 | 2018-03-02 | Móveis e decoração | Entregue no prazo | 1 |
| E3 | Prazo de entrega absurdo e demora no retorno do vendedor | 3 | 2018-06-30 | Brinquedos | Entregue no prazo | 1 |
| E4 | O produto chegou correto. O problema foi a demora na entrega. | 1 | 2017-12-12 | Brinquedos | Entregue no prazo | 1 |
| E5 | Um prazo de entrega demorado | 3 | 2017-08-19 | Artes | Entregue no prazo | 1 |
| E6 | Muita demora. O prazo de entrega foi longo demais | 3 | 2018-06-06 | Relógios e presentes | Entregue no prazo | 1 |
| E7 | demora na entrega (ainda nao chegou fora do prazo) | 1 | 2018-04-21 | Móveis e decoração | Entregue com atraso | 1 |
| E8 | Demora na entrega e ainda atrasou. | 3 | 2018-03-22 | Esporte e lazer | Entregue com atraso | 1 |
| E9 | O atraso na entrega prejudicou a obra | 3 | 2018-04-04 | Móveis e decoração | Entregue com atraso | 1 |
| E10 | Não é a primeira vez de atraso na entrega, quando consegue falar a atendente ainda pede um prazo de dois dias, fora o atraso, infelizmente as mudanças de entrega dos parceiros não está sendo pratica | 1 | 2018-03-09 | Utilidades domésticas | Entregue com atraso | 1 |

*review_id das evidências:* E1: 10467dc456e818c60bd750271b5d183e; E2: 60025b90e14804969d69eb764eb682a0; E3: 0cf3369f069c5f82123fe37c668d39b9; E4: 1da9941264ebc5c4a093ee4ba82af918; E5: cc7aa30132ae7195c15ac60a6d96f63d; E6: ba5c756c2d4d0d636ac63de5cedb771d; E7: a1e9b5056896b01638a5aa8e02673955; E8: 6964442cd4b5a5a0c5bbfa8d6fb914ef; E9: dd6e0f7168d75cd78769333f7630a036; E10: 6a24fdb69a1596a14f8ebe8386855576
*Tempo:* 1.4 s nesta execução · chamadas ao LLM: 2

---

### Quais são os principais problemas relatados nas avaliações com nota baixa?

**Decisão:** responder
**Recorte:** nota de 1 a 2 · 10.837 avaliações com texto

**Resposta:** Os principais problemas em avaliações com nota baixa (1 a 2) são a não entrega do produto (44%), defeitos ou má qualidade (18%) e pedidos incompletos (15%). As evidências exemplificam casos recorrentes de atrasos na entrega e produtos entregues com defeito ou ruins.

| Tema | Avaliações | % |
|---|---|---|
| Pedido não recebido | 4.797 | 44% |
| Defeito, produto errado ou má qualidade | 1.915 | 18% |
| Pedido incompleto, faltando itens | 1.588 | 15% |
| Atraso e prazo de entrega longo | 653 | 6% |
| Cor, tamanho ou aparência diferente do anúncio | 651 | 6% |
| Acessórios eletrônicos que não funcionam | 434 | 4% |

- No universo de 10.837 avaliações de nota 1 a 2, o problema mais frequente é 'Pedido não recebido' (44%), seguido de 'Defeito, produto errado ou má qualidade' (18%) e 'Pedido incompleto, faltando itens' (15%). [E2, E7, E9]
- Clientes relatam recebimento de produtos com defeito ou de péssima qualidade, além de dificuldades no atendimento ao tentar resolver o problema. [E1, E3, E4, E5, E6, E10]
- Há reclamações de atrasos na entrega e falta de comunicação sobre o status do pedido. [E1, E2, E7, E8, E9]

*Limitações:* As porcentagens apresentadas referem-se estritamente ao total de 10.837 avaliações com texto no recorte de nota 1 a 2. As evidências em texto são uma amostra qualitativa desse conjunto.

| Evidência | Avaliação do cliente | Nota | Data | Categoria | Entrega | Iguais |
|---|---|---|---|---|---|---|
| E1 | Atrasado e com defeito. Produto chegou atrasado, mal embalado e faltando um Ilhós. | 1 | 2018-07-21 | Bebês | Entregue no prazo | 1 |
| E2 | Atrasado. Receber com atraso é sempre ruim. O produto deveria ser entregue até 26/04/2018 e até o momento, não foi. | 2 | 2018-04-29 | Casa e construção | Entregue com atraso | 1 |
| E3 | produto com defeito. estou desde o dia 22 tentando resolver o problema do produto ter vindo com defeito, mas só me atendem no atendimento eletrônico. | 1 | 2018-08-23 | Cool stuff | Entregue no prazo | 1 |
| E4 | Produto Ruim. Relógio com Defeito | 2 | 2018-06-29 | Relógios e presentes | Entregue no prazo | 1 |
| E5 | Produto é muito ruim | 2 | 2017-12-23 | Brinquedos | Entregue no prazo | 1 |
| E6 | Produto já veio com defeito não funciona | 1 | 2018-08-02 | Relógios e presentes | Entregue no prazo | 1 |
| E7 | ruim. Estou preocupado, pois até agora não recebi o produto, e nem fui comunicado o porque está atrasado. | 1 | 2018-08-08 | Esporte e lazer | Cancelado ou indisponível | 1 |
| E8 | Atraso na entrega. ruim | 2 | 2018-03-30 | Informática e acessórios | Entregue no prazo | 1 |
| E9 | produto em atraso. | 1 | 2018-04-12 | Eletrônicos | Entregue com atraso | 1 |
| E10 | Produto com defeito. produto com defeito | 2 | 2018-05-04 | Beleza e saúde | Entregue no prazo | 1 |

*review_id das evidências:* E1: 6298846cdf0eceb7ba5661c3810725b7; E2: e057360360ac06fb9abc25efe4505413; E3: ced0560b692ae389904d584f03f4d8bf; E4: 217c82e96cff6b77a652bc921b7b7261; E5: 985c22a982d129b4b337a59f2d8f73a3; E6: 93f72220a2eefce8606c24ba7e121333; E7: 0159077dd9d4cb22186f4a0c65ac61da; E8: 55c854db53ac2dcb8eea576705c6fc19; E9: 02888ff454c764d781939ef65355e5a4; E10: eeafacf5860957c8c7f64dd182586816
*Tempo:* 1.7 s nesta execução · chamadas ao LLM: 2

---

### O que os clientes mais elogiam na categoria cama, mesa e banho?

**Decisão:** responder
**Recorte:** nota de 4 a 5; categoria: cama_mesa_banho · 2.762 avaliações com texto

**Resposta:** Os clientes da categoria cama, mesa e banho destacam principalmente o cumprimento do combinado em relação ao produto e à entrega (28%), elogios genéricos (25%) e pontualidade ou antecipação na entrega (23%).

| Tema | Avaliações | % |
|---|---|---|
| Produto e entrega conforme o combinado | 771 | 28% |
| Elogio curto e genérico | 702 | 25% |
| Entrega antes ou dentro do prazo | 633 | 23% |
| Satisfeito, com ressalvas | 237 | 9% |
| Entrega rápida | 142 | 5% |
| Beleza e acabamento do produto | 131 | 5% |

- O tema mais frequente é 'Produto e entrega conforme o combinado', representando 28% (771 avaliações) do total de 2.762 avaliações com texto do recorte. [E2, E3]
- Os temas relacionados ao cumprimento do prazo de entrega somam parcela expressiva, sendo 23% para 'Entrega antes ou dentro do prazo' (633 avaliações) e 5% para 'Entrega rápida' (142 avaliações). [E5]
- Nas avaliações qualitativas, os clientes elogiam frequentemente a qualidade do produto e sua estética/beleza, ressaltando a fidelidade às fotos exibidas. [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10]

*Limitações:* As porcentagens e contagens foram calculadas sobre o total de 2.762 avaliações do recorte (notas 4 a 5 na categoria cama_mesa_banho). As 10 evidências fornecidas servem como amostra das opiniões dos clientes.

| Evidência | Avaliação do cliente | Nota | Data | Categoria | Entrega | Iguais |
|---|---|---|---|---|---|---|
| E1 | super recomendo. produto maravilhoso, otima qualidade | 5 | 2018-06-23 | Cama, mesa e banho | Entregue no prazo | 1 |
| E2 | Produto de ótima qualidade, linda exatamente como na foto. Ficou perfeita. Amei. Recomendo a todos | 5 | 2018-03-21 | Cama, mesa e banho | Entregue no prazo | 1 |
| E3 | Excelente. Qualidade ótima, produto conforme a foto, lindo. Amei. | 5 | 2018-06-13 | Cama, mesa e banho | Entregue no prazo | 1 |
| E4 | Excelente. Muito lindo e combina perfeito com meu sofá,Adorei,eu recomendo esse produto porque a calidade é perfeita e ademais e muito bonito,com combinações de cores muito lindas,fico lindo em mi sala,eu amei. | 5 | 2018-06-12 | Cama, mesa e banho | Entregue no prazo | 1 |
| E5 | Muito Linda! Produto de ótima qualidade, ficou perfeita na minha cama. Chegou antes do prazo. | 5 | 2018-07-01 | Cama, mesa e banho | Entregue no prazo | 1 |
| E6 | Produto ótimo, de qualidade e muito bonito. Recomendadíssimo! | 4 | 2017-04-12 | Cama, mesa e banho | Entregue no prazo | 1 |
| E7 | Super recomendo produto de excelente qualidade. | 5 | 2017-06-06 | Cama, mesa e banho | Entregue no prazo | 1 |
| E8 | Ótimo produto! Linda cortina, delicada em perfeito estado! | 5 | 2017-05-20 | Cama, mesa e banho | Entregue no prazo | 1 |
| E9 | Produto de boa qualidade. Produto lindo e muito bom, ótima qualidade | 5 | 2018-04-27 | Cama, mesa e banho | Entregue no prazo | 1 |
| E10 | super recomendo 01092018. produto de ótima qualidade! | 5 | 2018-08-29 | Cama, mesa e banho | Entregue no prazo | 1 |

*review_id das evidências:* E1: adeb4cd136b9f49ba7f7e037ad87cf99; E2: 9dd0d1667176b379cf2b2e1452e863b7; E3: 39262799606cd86e7715a463703ff241; E4: b9e63bee318128950eb50d7b82b8e76d; E5: 93af3ee75114688b4e0fb0721ee74b97; E6: e756e7d72459f513ac8cbf720f1e56cb; E7: de6a12da73417b15a0a4d6f3a7e41251; E8: 62b805f4bc0c1ef150bf16abc593cb3f; E9: 2c9e8b76b29a327017a7d060a2324af3; E10: 57e2e1fa98b55f90e0999a8620a3cf1f
*Tempo:* 1.5 s nesta execução · chamadas ao LLM: 2

---

### Quais temas aparecem nas avaliações de entregas atrasadas?

**Decisão:** responder
**Recorte:** entregues com atraso; entrega: atrasada · 3.814 avaliações com texto

**Resposta:** Nas avaliações de pedidos entregues com atraso, os principais temas são o não recebimento do produto e o atraso ou prazo longo de entrega, além de queixas sobre qualidade, descaso do vendedor e alguns casos pontuais de clientes satisfeitos ou que receberam com ressalvas.

| Tema | Avaliações | % |
|---|---|---|
| Pedido não recebido | 1.987 | 52% |
| Atraso e prazo de entrega longo | 380 | 10% |
| Defeito, produto errado ou má qualidade | 334 | 9% |
| Entrega antes ou dentro do prazo | 175 | 5% |
| Produto e entrega conforme o combinado | 160 | 4% |
| Elogio curto e genérico | 154 | 4% |

- O tema mais frequente é 'Pedido não recebido', representando 52% (1.987 avaliações) do conjunto de 3.814 avaliações do recorte. [E1, E2]
- O tema 'Atraso e prazo de entrega longo' responde por 10% (380 avaliações) do total analisado. [E3, E4, E5, E6, E7, E8, E9, E10]
- Outros temas identificados incluem 'Defeito, produto errado ou má qualidade' (9%), 'Entrega antes ou dentro do prazo' (5%), 'Produto e entrega conforme o combinado' (4%), 'Elogio curto e genérico' (4%), 'Satisfeito, com ressalvas' (4%) e 'Descaso da loja com o cliente' (3%). [E4, E5, E9]

*Limitações:* A tabela de frequência engloba o universo de 3.814 avaliações com texto do recorte, porém os exemplos textuais recuperados como evidências abordam predominantemente o atraso, não recebimento e falhas de comunicação da loja.

| Evidência | Avaliação do cliente | Nota | Data | Categoria | Entrega | Iguais |
|---|---|---|---|---|---|---|
| E1 | entrega atrasada. o prazo terminou e nao recebi meu produto | 1 | 2018-06-01 | Informática e acessórios | Entregue com atraso | 1 |
| E2 | demora na entrega (ainda nao chegou fora do prazo) | 1 | 2018-04-21 | Móveis e decoração | Entregue com atraso | 1 |
| E3 | Passou do prazo de entrega, demorou. | 1 | 2018-01-18 | Esporte e lazer | Entregue com atraso | 1 |
| E4 | Entrega demorada. A encomenda demorou muito a chegar mas chegou em perfeito estado. | 4 | 2018-06-01 | Relógios e presentes | Entregue com atraso | 1 |
| E5 | não cumpre prazo. Não cumpriram prazo de entrega e nem avisaram o atraso. | 1 | 2018-08-12 | Alimentos | Entregue com atraso | 1 |
| E6 | atraso na entrega da mercadoria,não cumpre prazo determinado. | 1 | 2018-08-24 | Cama, mesa e banho | Entregue com atraso | 1 |
| E7 | Atraso. Não foi entregue no prazo estimado! | 1 | 2018-08-23 | Perfumaria | Entregue com atraso | 1 |
| E8 | Demora na entrega e ainda atrasou. | 3 | 2018-03-22 | Esporte e lazer | Entregue com atraso | 1 |
| E9 | prazo de entrega ultrapassou o previsto, vi pelo status q ocorreu algo durante o transporte da mercadoria e esperava um contato para justificar o acontecido e q por isso a entrega atrasaria mas ñ tive | 1 | 2017-04-13 | Consoles e games | Entregue com atraso | 1 |
| E10 | Demorou de mais pra chegar, se tivesse comprado fora do país não demoraria tanto. | 2 | 2018-01-20 | Cool stuff | Entregue com atraso | 1 |

*review_id das evidências:* E1: 1bfec3e913196c72c1ea06fc8816bd9c; E2: a1e9b5056896b01638a5aa8e02673955; E3: 2c74e9b846eeb175c36f4e5fd9f92eb3; E4: 1faf54dfeb5d82e5218a8ca9433e90b8; E5: 85a391b84d8e864de7190a2afb6f82e4; E6: 10467dc456e818c60bd750271b5d183e; E7: ad643ac0f211566061c0f5a39d403a51; E8: 6964442cd4b5a5a0c5bbfa8d6fb914ef; E9: b72948c0912824f5660a661fcb3ef512; E10: 80457ae14ae95b6b2cca6ce4bffb123a
*Tempo:* 1.7 s nesta execução · chamadas ao LLM: 2

---

### Qual foi o faturamento total da Olist em 2018?

**Decisão:** recusar (fora_do_escopo)
**Recorte:** compra de 2018-01 a 2018-12

**Resposta:** Não é possível responder com as avaliações de clientes: a pergunta pede uma informação que elas não contêm (métricas internas, financeiras ou assuntos alheios à experiência de compra). A base reúne notas e comentários de clientes sobre pedidos da Olist feitos entre 2016 e 2018: entrega, produto, atendimento, loja e elogios.



*Tempo:* 0.0 s nesta execução · chamadas ao LLM: 1

---

### O que os clientes acham do pagamento via PIX?

**Decisão:** recusar (evidencia_insuficiente)
**Recorte:** sem filtro · 42.093 avaliações com texto

**Resposta:** As avaliações encontradas não tratam do assunto perguntado com relevância suficiente para sustentar uma resposta. Não há evidência nos comentários para afirmar algo sobre isso.



*Tempo:* 1.8 s nesta execução · chamadas ao LLM: 1

---

### O que os clientes falam dos produtos da categoria seguros e serviços?

**Decisão:** recusar (poucas_avaliacoes)
**Recorte:** categoria: seguros_e_servicos · 2 avaliações com texto

**Resposta:** Há só 2 avaliação(ões) com comentário no recorte pedido (categoria: seguros_e_servicos), poucas para identificar um padrão.



*Tempo:* 0.3 s nesta execução · chamadas ao LLM: 1
