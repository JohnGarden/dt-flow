# Decision Transformers for Flow-based Intrusion Detection on UNSW-NB15

Este repositório contém o artefato associado ao artigo submetido ao SBRC 2026, que investiga o uso de Decision Transformers (DT) para detecção de intrusões em tráfego de rede, tratando fluxos do dataset UNSW-NB15 como trajetórias temporais em um cenário de aprendizado por reforço offline.

O artefato disponibiliza:

- código-fonte para pré-processar o UNSW-NB15;
- implementação do Decision Transformer para classificação binária (normal vs. ataque);
- baselines clássicos (Isolation Forest, RNN, Transformer NIDS);
- scripts para reprodução das principais tabelas e resultados do artigo.

**Título do artigo:**  
> *Decision Transformers for Flow-based Intrusion Detection on UNSW-NB15: A Reproducible Study*

**Resumo do artigo (em inglês):**  
This paper investigates network intrusion detection as a sequence modeling problem in offline reinforcement learning. We propose a Decision Transformer-based IDS that treats UNSW-NB15 flows as temporal trajectories and augments tabular features with inter-arrival time embeddings. Rewards encode the cost asymmetry between false positives and false negatives, and the model predicts intrusion labels conditioned on returns-to-go. Evaluated on binary normal-vs-attack classification against Isolation Forest, recurrent networks and Transformer-based NIDS, the approach attains competitive or superior F1-scores under class imbalance, indicating that RL-inspired sequence modeling is promising for intrusion detection.

**Resumo do artigo (em português):**  
Este artigo investiga a detecção de intrusões em redes como um problema de modelagem de sequências em aprendizado por reforço offline. Propomos um IDS baseado em Decision Transformer que trata os fluxos do UNSW-NB15 como trajetórias temporais e enriquece as features tabulares com embeddings de tempo entre chegadas. As recompensas codificam a assimetria de custo entre falsos positivos e falsos negativos, e o modelo prediz rótulos de intrusão condicionado ao retorno futuro desejado. Avaliado em classificação binária normal versus ataque, em comparação com Isolation Forest, redes recorrentes e NIDS baseados em Transformer, o método atinge F1-scores competitivos ou superiores em cenários desbalanceados, indicando que a modelagem de sequências inspirada em RL é uma abordagem promissora para detecção de intrusões.

---

## Estrutura do README.md

Este README está organizado nas seguintes seções, conforme as instruções da avaliação de artefatos:

1. **Título projeto** - contextualização do artefato e do artigo.
2. **Estrutura do readme.md** - organização do repositório.
3. **Selos Considerados** - selos de qualidade de artefato pleiteados.
4. **Informações básicas** - ambiente, requisitos de hardware e software.
5. **Dependências** - bibliotecas, ferramentas e benchmarks/datasets.
6. **Preocupações com segurança** - riscos e recomendações.
7. **Instalação** - passo a passo para preparar o ambiente.
8. **Teste mínimo** - verificação rápida de funcionamento.
9. **Experimentos** - reprodução das principais reivindicações do artigo.
10. **Estrutura do Código** - organização do código-fonte.
11. **LICENSE** - informações de licença.

---

## Selos Considerados

Os selos de qualidade de artefato considerados para este repositório são:

- **Artefatos Disponíveis (SeloD)**  
  Código-fonte e scripts públicos neste repositório (GitHub).
- **Artefatos Funcionais (SeloF)**  
  Instruções para instalação, execução, dependências e teste mínimo.
- **Artefatos Sustentáveis (SeloS)**  
  Organização modular do código, documentação básica e mapeamento claro com as reivindicações do artigo.
- **Experimentos Reprodutíveis (SeloR)**  
  Passo a passo para reproduzir as principais tabelas e métricas reportadas no artigo, incluindo scripts automatizados.

---

## Informações básicas

### Ambiente de execução recomendado

- **Sistema operacional**  
  - Windows 11 64 bits (testado em ambiente de desenvolvimento)
- **Versão de Python**  
  - Python 3.10 ou superior
- **Hardware**  
  - CPU: Intel/AMD com suporte a 64 bits  
  - Memória RAM: mínimo 16 GB (recomendado 32 GB para treino completo)  
  - GPU (opcional, mas recomendado):  
    - GPU NVIDIA com pelo menos 8 GB de VRAM (ex.: GeForce RTX 3070 8 GB)  
    - Driver NVIDIA e CUDA instalados (CUDA 12.x ou compatível com a versão do PyTorch)
- **Espaço em disco**  
  - ~10–20 GB livres para:  
    - CSVs do UNSW-NB15,  
    - artefatos de treino,  
    - resultados de experimentos.

### Tempo aproximado de execução

Em uma máquina com:  

- CPU **Intel Core i7-11800H**,  
- **32 GB RAM**,  
- **NVIDIA GeForce RTX 3070 (8 GB)**,

observa-se, de forma aproximada:

- **Teste mínimo** (sanity check): < 1 minutos.  
- **Treino completo do Decision Transformer** (`run_dt_baseline.sh`): < 5 minutos.  
- **Treino dos baselines + avaliação de métricas**: < 3 minutos.

---

## Dependências

### Linguagem e ferramentas

- **Python** ≥ 3.10
- **Git** (para clonar o repositório)
- **virtualenv** ou suporte nativo a ambientes virtuais do Python (`venv`)
- **CUDA/cuDNN** (opcional, apenas para uso de GPU com PyTorch)

### Bibliotecas principais (Python)

A maioria das dependências é instalada automaticamente via `scripts/setup_env.sh`, mas, em alto nível, o projeto utiliza:

- **PyTorch** (CPU ou GPU) - `torch`, `torchvision`, `torchaudio`
- **NumPy**, **Pandas**
- **scikit-learn**
- **PyYAML**
- **tqdm**
- Outras bibliotecas auxiliares especificadas em `requirements.txt`.

Recomenda-se realizar a  instalação manual do CUDA diretamente no [site oficial](https://developer.nvidia.com/cuda-downloads).

### Benchmark / Dataset

- **UNSW-NB15** - dataset público de detecção de intrusões em rede.
  - Página oficial: [The UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- O projeto assume que os CSVs do UNSW-NB15 (treino/teste) serão copiados para `data/raw/unsw-nb15/csv/` (detalhado na seção de Instalação).

---

## Preocupações com segurança

- O artefato trabalha **apenas com dados offline** (CSVs do UNSW-NB15) e **não realiza qualquer varredura ativa** na rede do avaliador.
- Não há código para enviar pacotes, abrir portas, executar exploits ou interagir com sistemas externos.
  Toda a lógica é de processamento de dados e treinamento/inferência de modelos.
- O dataset UNSW-NB15 contém **tráfego rotulado como malicioso**, mas somente em forma de registros tabulares; não há execução de código malicioso.
- Recomendações:

  - Executar o artefato em um ambiente isolado (ex.: máquina virtual ou ambiente conda/venv dedicado).
  - Evitar rodar o código como `root`/administrador sem necessidade.

---

## Instalação

Abaixo um passo a passo para preparar o ambiente do artefato.

### 1. Clonar o repositório

```bash
git clone https://github.com/JohnGarden/dt-flow.git
cd dt-flow
```

### 2. Criar e ativar ambiente virtual

```bash
bash scripts/setup_env.sh
```

O script `scripts/setup_env.sh` deve:

- criar o ambiente virtual `.venv/`;
- instalar as dependências listadas.

### 3. Preparar diretório de dados e copiar UNSW-NB15

#### 1. Crie a estrutura de diretórios esperada

   ```bash
   bash scripts/prepare_unsw.sh
   ```

#### 2. Baixe os arquivos CSV do UNSW-NB15 a partir da página oficial ou utilize a versão disponível neste repositório

- Página oficial: [The UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

#### 3. Copie os arquivos CSV para

   ```text
   data/raw/unsw-nb15/csv/
   ```

### 4. Sanity check

Execute um teste rápido para verificar se:

- o ambiente está configurado;
- o dataset foi encontrado;
- as configs são válidas.

```bash
bash scripts/run_sanity.sh
```

Este script deve rodar em poucos minutos e imprimir logs básicos de carregamento de dados e configuração do modelo.

---

## Teste mínimo

O **teste mínimo** consiste em:

1. Criar e ativar o ambiente virtual.
2. Preparar a estrutura de dados.
3. Rodar o sanity check.

Comandos completos (Linux/WSL):

```bash
# dentro do repositório
bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/prepare_unsw.sh
bash scripts/run_sanity.sh
```

Comandos completos (Windows):

```bash
# dentro do repositório
bash scripts/setup_env.sh
source .venv/Scripts/activate
bash scripts/prepare_unsw.sh
bash scripts/run_sanity.sh
```

Resultados esperados:

- o script deve:
  - importar corretamente as dependências;
  - localizar os arquivos CSV do UNSW-NB15;
  - descobre o Python da venv;
  - garante que o pacote do projeto (`im12dt`) é importável;
  - verificar se os arquivos de configuração YAML estão presentes e bem formados;

- tempo de execução: em geral **< 1 minuto** na máquina recomendada.

Se este teste passar, podemos executar os experimentos completos descritos a seguir.

---

## Experimentos

Esta seção descreve como reproduzir as principais reivindicações do artigo. Cada reivindicação é organizada em uma subseção, com comandos e recursos esperados.

### Reivindicação #1 - Desempenho do Decision Transformer em classificação binária (Tabela 1)

**Descrição:**
O Decision Transformer (DT) treinado sobre o UNSW-NB15 (normal vs. ataque) atinge métricas de classificação binária (Accuracy, Precision, Recall, F1) superiores às de:

- Isolation Forest,
- RNN (GRU/LSTM),
- Transformer NIDS supervisionado.

#### 1. **Ativar ambiente e garantir dados**

```bash
cd dt-flow/
source .venv/Scripts/activate    # SO Windows
```

#### 2. **Treinar o Decision Transformer baseline**

```bash
bash scripts/run_dt_baseline.sh
```

- Este script:

  - lê `configs/data.yaml`, `configs/model_dt.yaml` e `configs/trainer.yaml`;
  - prepara o dataset sequencial;
  - treina o Decision Transformer;
  - salva um checkpoint em `artifacts/dt_baseline_YYYYMMDD-HHMMSS.pt`.

- **Recursos esperados:**

  - GPU com 8 GB de VRAM é recomendada, mas CPU também é possível.
  - Tempo de execução: até 2 minutos na máquina de referência (i7-11800H + RTX 3070).

#### 3. **Treinar baselines (Isolation Forest, RNN, Transformer NIDS)**

```bash
bash scripts/run_baselines_unsw.sh
```

- Este script deve:

  - treinar o Isolation Forest (usando apenas flows normais para treino);
  - treinar RNN (GRU/LSTM) supervisionada;
  - treinar Transformer NIDS supervisionado;
  - salvar logs/artefatos em `artifacts/benchmark/`.

#### 4. **Resultado esperado**

- O Decision Transformer deve apresentar:

  - **F1-score** e **Recall** superiores aos baselines, em especial em comparação ao Isolation Forest e RNN, no cenário binário normal vs. ataque.
- Pequenas diferenças numéricas em relação ao artigo são aceitáveis, desde que a **tendência relativa** (DT > Transformer NIDS > RNN > IF) seja mantida.

#### 5. Recursos e tempo #1

- CPU + GPU: a mesma configuração usada no treino é suficiente.
- Tempo aproximado: < 5 minutos.

---

### Reivindicação #2 - Comparação com baselines

**Descrição:**
O DT explora a formulação de decisão sequencial (recompensas + return-to-go) e trajetória temporal (host + janelas + $\Delta t_t$) para atingir desempenho superior ou competitivo em relação a IF, RNN e Transformer NIDS treinados apenas como classificadores supervisionados.

Esta reivindicação utiliza os mesmos comandos da **Reivindicação #1**.

A partir dos arquivos abaixo:

- `artifacts/benchmark/metrics_isolation_forest.csv`
- `artifacts/benchmark/metrics_rnn.csv`
- `artifacts/benchmark/metrics_transformer.csv`
- `artifacts/ttr_grid_any_step.csv`

O avaliador pode:

- comparar diretamente as métricas de IF vs. RNN vs. Transformer vs. DT;
- confirmar que o DT atinge:

  - maior F1-score;
  - desempenho semelhante ou superior ao Transformer NIDS.

#### Recursos e tempo #2

- Não há passos adicionais de execução além dos já descritos na Reivindicação #1.
- Tempo adicional é apenas o de análise dos valores no CSV.

---

### Reivindicação #3 - Avaliação de latência (TTR) e política any-step

**Descrição:**
Dado um Decision Transformer treinado, é possível avaliar o comportamento do modelo em um cenário quase tempo real, medindo Time-To-Resolution (TTR) sob políticas de alarme como **any-step** (gera alerta se qualquer passo na trajetória for ataque).

#### 1. **Ativar ambiente e garantir checkpoint**

```bash
cd dt-flow
source .venv/Scripts/activate
```

> Certifique-se de que `run_dt_baseline.py` foi executado e que existe um checkpoint em `artifacts/dt_baseline_*.pt`.

#### 2. **Executar avaliação de TTR**

```bash
python scripts/run_ttr_eval.py
```

- Este script deve:

  - carregar o modelo DT treinado;
  - executar a avaliação em modo streaming / janela;
  - salvar os resultados em arquivos CSV dentro de `artifacts/ttr_eval/`.

#### 3. Recursos e tempo #3

- CPU + GPU: a mesma configuração usada no treino é suficiente.
- Tempo típico: < 3 minutos.

---

### Reivindicação #4 - Treinamento multi-seed

**Descrição:**
É possível treinar o Decision Transformer diversas vezes com sementes aleatórias diferentes, de forma automatizada, para avaliar a robustez estatística do modelo e gerar um conjunto de checkpoints identificados por semente. Isso é feito pelo script `run_multiseed.py`, que chama internamente o `run_dt_baseline.py`, ajusta a semente global e opcionalmente sobrescreve o número de épocas e passos por época.

---

#### 1. **Ativar ambiente para multi-seed**

```bash
cd dt-flow
source .venv/Scripts/activate
```

---

#### 2. **Executar treino multi-seed**

Exemplo com três sementes (42, 43, 44):

```bash
python scripts/run_multiseed.py --seeds "42,43,44"
```

- Este comando deve:

  - definir, para cada semente:

    - `random.seed`, `numpy.random.seed` e `torch.manual_seed`, garantindo reprodutibilidade por semente;
  - chamar o `run_dt_baseline.main()` uma vez por semente, lendo as mesmas configs YAML do projeto;
  - detectar o novo checkpoint gerado em `artifacts/` após cada execução;
  - renomear o checkpoint para incluir a semente, por exemplo:
    `dt_baseline_20251110-221610_seed42.pt`;
  - repetir o processo para todas as sementes da lista e imprimir, ao final, o caminho de cada checkpoint gerado.

#### 2. **Recursos e tempo #4**

- CPU + GPU: a mesma configuração usada no treino é suficiente.
- Tempo típico: < 5  minutos.

---

### Reivindicação #5 - Calibração de limiares (class_cut e wait_threshold) com TTR

**Descrição:**
Dado um Decision Transformer já treinado, é possível executar uma **varredura sistemática de limiares** de classificação (`class_cut`) e de confiança para disparo de alarme (`wait_threshold`), avaliando simultaneamente **F1 token-level** e **Time-To-Resolution (TTR)** nas políticas *last-step*, *any-step* e *tail m-of-last L*, sem precisar rodar o modelo várias vezes. O script `run_calibration.py` faz uma única passada no conjunto de teste, guarda os tensores de saída e, a partir deles, calcula todas as combinações de métricas e gera CSVs de calibração.

---

#### 1. **Ativar ambiente para calibração**

```bash
cd dt-flow
source .venv/Scripts/activate
```

> Certifique-se de que o modelo DT já foi treinado e que existe pelo menos um checkpoint em `artifacts/dt_baseline_*.pt`.

---

#### 2. **Executar calibração de limiares**

Calibração padrão, usando o checkpoint mais recente em `artifacts/`:

```bash
python scripts/run_calibration.py
```

- Este script deve:

  - carregar arquivos de configuração;
  - reconstruir o dataset sequencial de teste (`UNSWSequenceDataset`);
  - reconstruir `StateTokenizer`, `ActionTokenizer`, `RTGTokenizer`, `TimeEncodingFourier` e o `DecisionTransformer`;
  - fazer **uma única passada** no conjunto de teste, acumulando:

    - tensores token-level: `y` (rótulos), `p` (prob. de ataque), `m` (máscara válida);
    - estatísticas por fluxo: tempos de primeiro ataque verdadeiro, predições por *last-step*, *any-step* e *tokens de cauda* (tail);
  - varrer a grade de `class_cut` (`--grid-cut`) para:

    - calcular PR-AUC (único) e F1 para cada limiar de probabilidade;
    - identificar o melhor `class_cut` em termos de F1;
  - varrer a grade de `wait_threshold` (`--grid-wait`) para:

    - calcular, em cada política de TTR (*last-step*, *any-step*, *tail m-of-last L*):

      - nº de fluxos maliciosos (`flows_plus`);
      - nº de fluxos detectados;
      - taxa de detecção (`rate`);
      - TTR_P50, TTR_P90, TTR_avg, TTR_max;
  - gerar arquivos em `artifacts/calibration_<timestamp>/` (ou `--outdir`), incluindo:

    - `token_f1_grid.csv` (F1 vs `class_cut`);
    - `ttr_grid_last_step.csv`, `ttr_grid_any_step.csv`, `ttr_grid_tail_mL.csv` (TTR vs `wait_threshold`);
    - `calibration_summary.txt` com:

      - PR-AUC token-level;
      - melhor `class_cut` em F1;
      - melhores `wait_threshold` para cada política de TTR segundo a regra:

        - se `rate >= 0.99`, escolher menor TTR_P90;
        - caso contrário, escolher maior `rate`.

---

#### 3. Recursos e tempo #5

- CPU + GPU: a mesma configuração usada no treino é suficiente.
- Tempo típico: < 3 minutos.

---

## Estrutura do Código

O código-fonte está organizado da seguinte forma:

```text
- artifacts/ - checkpoints do Decision Transformer (`dt_baseline_*.pt`).
  - benchmark/ - resultados comparativos com IF, RNN, Transformer.
  - calibration_YYYYMMDD-HHMMSS/ - arquivos resultantes da calibração dos limiares.
  - ttr_eval/ - resultados DT.
- configs/ - arquivos YAML de configuração (dados, modelo, treino).
- data/  
  - raw/unsw-nb15/csv/ - CSVs originais do UNSW-NB15.  
- scripts/ - scripts auxiliares (setup, preparação de dados, baselines, avaliação).
- src/
  - im12dt/ - código-fonte do projeto.
- pyproject.toml - definição do pacote Python (`im12dt`).
- README.md - este documento.
```

## LICENSE

Este projeto está licenciado sob os termos da licença **MIT**.
Consulte o arquivo [`LICENSE`](LICENSE) na raiz deste repositório para mais detalhes.
