# Filtros_TIP8544
trabalho 2 disciplina Técnicas de controle digital
# Análise Comparativa de Filtros Analógicos Ativos e Passivos
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rnunesufc/Filtros_TIP8544/blob/main/Trabalho2.ipynb)


## 🎯 Objetivo do Projeto

Este repositório contém um estudo comparativo de topologias clássicas de filtros passa-baixa analógicos. O objetivo é analisar a teoria, implementar simulações em Python e visualizar os *trade-offs* (trocas) de engenharia entre diferentes famílias de filtros.

Toda a análise e apresentação dos resultados estão consolidadas no notebook Jupyter: **`Trabalho2.ipynb`**.

## 🔬 Filtros Analisados

Este estudo implementa e compara 6 tipos de filtros, todos projetados para uma frequência de corte alvo de $\omega_c \approx 2000 \text{ rad/s}$:

1.  **Filtro RC (1ª Ordem)**: A base passiva.
2.  **Filtro RC-RC (2ª Ordem)**: Filtro passivo em cascata (polos reais).
3.  **Filtro Butterworth (N=2)**: Ativo (Sallen-Key), maximamente plano.
4.  **Filtro Chebyshev (N=2)**: Ativo (Sallen-Key), otimizado para *roll-off* (com *ripple*).
5.  **Filtro Cauer/Elíptico (N=2)**: Ativo (Biquad), otimizado para *roll-off* (com *ripple* e *notches*).
6.  **Filtro Bessel (N=2)**: Ativo (Sallen-Key), otimizado para o domínio do tempo (fase linear).

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma para separar a apresentação dos scripts-fonte:
/ ├── Trabalho2.ipynb <-- O notebook principal com toda a análise ├── scripts/ │ ├── filtro1aOrdem.py │ ├── filtro2aOrdem.py │ ├── Butterworth2aOrdem.py │ ├── Chebt1.py │ ├── Cauer_state_var.py │ └── Bessel_2.py ├── Fig/ │ ├── sallen_key.png │ └── biquad.png └── README.md <-- Este arquivo


* **`Trabalho2.ipynb`**: O relatório final. É aqui que a teoria é explicada e os resultados são apresentados.
* **`scripts/`**: Contém os scripts Python independentes para cada filtro. Cada script é uma "caixa-preta" que gera a FT, calcula os componentes e plota os 4 gráficos de análise (Tempo, FFT, Bode, Polos/Zeros).
* **`Fig/`**: Contém as imagens estáticas dos diagramas de circuito (Sallen-Key, Biquad) que são usadas nos textos de teoria do notebook.

## ⚙️ Como Funciona (O Fluxo do Notebook)

O notebook `Trabalho2.ipynb` foi projetado para ser um relatório interativo e reprodutível. Ele segue um fluxo de trabalho específico:

1.  **Setup Inicial**: As primeiras células importam as bibliotecas (`numpy`, `matplotlib`, `scipy`, `control`) e definem a "bancada de testes" (o sinal de entrada `U_A_ruido` com 60Hz + 1000Hz).

2.  **Execução por Capítulo**: Para cada filtro, o notebook possui duas células:
    * **Célula de Texto (Markdown)**: Explica a teoria, os *trade-offs* e mostra o diagrama do circuito relevante.
    * **Célula de Código (Code)**: Usa o comando mágico `%run` para executar o script correspondente da pasta `scripts/`. (ex: `%run scripts/Butterworth2aOrdem.py`).

3.  **Geração e Captura**:
    * Ao ser executado, o script `.py` gera e exibe seus próprios gráficos (Tempo, FFT, Bode) diretamente na saída da célula.
    * **Crucialmente:** Imediatamente após a execução, a célula "captura" as variáveis de resultado (como `w`, `mod`, `y_out_A_ruido`) do escopo global e as salva em variáveis com nomes únicos e permanentes (ex: `w_butter`, `mag_butter`, `y_out_A_ruido_butter`).

4.  **Comparação Final**:
    * Após todos os filtros serem executados, o notebook usa todas as variáveis capturadas (`w_butter`, `w_cheby`, `w_cauer`, etc.) para gerar os gráficos de **Comparação Final** (Células 17 e 19).
    * Esses gráficos finais plotam todos os Diagramas de Bode e todas as respostas no Domínio do Tempo juntas, permitindo uma visualização clara dos *trade-offs* de engenharia.

## 🐍 Dependências

Para executar este notebook, você precisará das seguintes bibliotecas Python:

* `jupyter`
* `numpy`
* `matplotlib`
* `scipy`
* `python-control`
