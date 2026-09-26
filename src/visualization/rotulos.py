"""Rotulos de negocio para graficos e tabelas.

Os graficos usam estes nomes no lugar dos identificadores tecnicos, para que um
gestor leia o eixo sem consultar o dicionario de dados.
"""

from __future__ import annotations

CATEGORY_LABELS: dict[str, str] = {
    "agro_industria_e_comercio": "Agroindústria e comércio",
    "alimentos": "Alimentos",
    "alimentos_bebidas": "Alimentos e bebidas",
    "artes": "Artes",
    "artes_e_artesanato": "Artes e artesanato",
    "artigos_de_festas": "Artigos de festas",
    "artigos_de_natal": "Artigos de Natal",
    "audio": "Áudio",
    "automotivo": "Automotivo",
    "bebes": "Bebês",
    "bebidas": "Bebidas",
    "beleza_saude": "Beleza e saúde",
    "brinquedos": "Brinquedos",
    "cama_mesa_banho": "Cama, mesa e banho",
    "casa_conforto": "Casa e conforto",
    "casa_conforto_2": "Casa e conforto 2",
    "casa_construcao": "Casa e construção",
    "cds_dvds_musicais": "CDs e DVDs musicais",
    "cine_foto": "Cinema e foto",
    "climatizacao": "Climatização",
    "consoles_games": "Consoles e games",
    "construcao_ferramentas_construcao": "Ferramentas de construção",
    "construcao_ferramentas_ferramentas": "Ferramentas",
    "construcao_ferramentas_iluminacao": "Iluminação",
    "construcao_ferramentas_jardim": "Ferramentas de jardim",
    "construcao_ferramentas_seguranca": "Ferramentas de segurança",
    "cool_stuff": "Cool stuff",
    "dvds_blu_ray": "DVDs e Blu-ray",
    "eletrodomesticos": "Eletrodomésticos",
    "eletrodomesticos_2": "Eletrodomésticos 2",
    "eletronicos": "Eletrônicos",
    "eletroportateis": "Eletroportáteis",
    "esporte_lazer": "Esporte e lazer",
    "fashion_bolsas_e_acessorios": "Bolsas e acessórios",
    "fashion_calcados": "Calçados",
    "fashion_esporte": "Moda esportiva",
    "fashion_roupa_feminina": "Roupa feminina",
    "fashion_roupa_infanto_juvenil": "Roupa infantojuvenil",
    "fashion_roupa_masculina": "Roupa masculina",
    "fashion_underwear_e_moda_praia": "Moda íntima e praia",
    "ferramentas_jardim": "Ferramentas e jardim",
    "flores": "Flores",
    "fraldas_higiene": "Fraldas e higiene",
    "industria_comercio_e_negocios": "Indústria e comércio",
    "informatica_acessorios": "Informática e acessórios",
    "instrumentos_musicais": "Instrumentos musicais",
    "la_cuisine": "La Cuisine",
    "livros_importados": "Livros importados",
    "livros_interesse_geral": "Livros de interesse geral",
    "livros_tecnicos": "Livros técnicos",
    "malas_acessorios": "Malas e acessórios",
    "market_place": "Marketplace",
    "moveis_colchao_e_estofado": "Colchões e estofados",
    "moveis_cozinha_area_de_servico_jantar_e_jardim": "Móveis de cozinha e jardim",
    "moveis_decoracao": "Móveis e decoração",
    "moveis_escritorio": "Móveis de escritório",
    "moveis_quarto": "Móveis de quarto",
    "moveis_sala": "Móveis de sala",
    "musica": "Música",
    "papelaria": "Papelaria",
    "pc_gamer": "PC gamer",
    "pcs": "Computadores",
    "perfumaria": "Perfumaria",
    "pet_shop": "Pet shop",
    "portateis_casa_forno_e_cafe": "Fornos e cafeteiras",
    "portateis_cozinha_e_preparadores_de_alimentos": "Portáteis de cozinha",
    "relogios_presentes": "Relógios e presentes",
    "seguros_e_servicos": "Seguros e serviços",
    "sinalizacao_e_seguranca": "Sinalização e segurança",
    "tablets_impressao_imagem": "Tablets e impressão",
    "telefonia": "Telefonia",
    "telefonia_fixa": "Telefonia fixa",
    "utilidades_domesticas": "Utilidades domésticas",
}

DELIVERY_LABELS = {
    "no prazo": "Entregue no prazo",
    "atrasada": "Entregue com atraso",
    "nao entregue": "Não entregue",
    "cancelada": "Cancelado ou indisponível",
}

SCORE_LABELS = {1: "1 estrela", 2: "2 estrelas", 3: "3 estrelas", 4: "4 estrelas",
                5: "5 estrelas"}

COLUMN_LABELS = {
    "review_score": "Nota da avaliação",
    "has_comment": "Escreveu comentário",
    "has_title": "Escreveu título",
    "delivery_status": "Situação da entrega",
    "delay_days": "Dias de atraso",
    "delivery_days": "Dias até a entrega",
    "main_category": "Categoria principal",
    "customer_region": "Região do cliente",
    "purchase_month": "Mês da compra",
    "answered_before_delivery": "Respondeu antes da entrega",
}


def category_label(name: str | None) -> str:
    """Nome da categoria para leitura; categorias novas caem no formato generico."""
    if name is None or name != name:
        return "Sem categoria"
    return CATEGORY_LABELS.get(name, name.replace("_", " ").capitalize())


def label(column: str) -> str:
    return COLUMN_LABELS.get(column, column)
