# Filtros_TIP8544
trabalho 2 disciplina Técnicas de controle digital
# Análise Comparativa de Filtros Analógicos Ativos e Passivos

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
