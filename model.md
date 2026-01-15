flowchart LR
  %% =======================
  %% Left: Text/Graph branch
  %% =======================
  subgraph T["(1) LLM-Augmented Semantic Graph (Text Branch)"]
    A0["Discrete primitives: A (attributes), O (objects)"] --> LLM["LLM (DeepSeek-V3.1)\nEvidence E_i={def,hyper,syn,hyponym}"]
    LLM --> BERT["Frozen BERT encoder\n(evidence embeddings)"]
    BERT --> CLU["Hierarchical clustering\n+ prototype pooling"]
    CLU --> H0["Init concept emb: h_i^(0)"]
    H0 --> G["Hetero Graph G=(V_a∪V_o, E_aa,E_ao,E_oo)\nTop-k sim + co-occur (E_ao)"]
    G --> HGNN["HGNN message passing (L layers)"]
    HGNN --> HA["Graph-enhanced attr emb: h_a"]
    HGNN --> HO["Graph-enhanced obj emb: h_o"]
    HA --> ZG["Word-side composition\nz_g(a,o)=g_p^w([h_a||h_o])"]
    HO --> ZG
  end

  %% =======================
  %% Right: Visual branch
  %% =======================
  subgraph V["(2) VSEM (Visual Branch)"]
    X["Image x"] --> ViT["ViT backbone\nf_v(x)"]
    ViT --> PA["Attr encoder ϕ_a (MLP)\nf_v^a(x)"]
    ViT --> PO["Obj encoder ϕ_o (MLP)\nf_v^o(x)"]
    PA --> ZA["Projection W^a\nz_v^a"]
    PO --> ZO["Projection W^o\nz_v^o"]
    ZA --> ZP["Visual-side composition\nz_v^p=g_p^v([z_v^a||z_v^o])"]
    ZO --> ZP

    ViT --> DPM["DPM heads\nCE over A and O"]
    ZA --> ISa["IS (attr)\nInfoNCE on g_a(z_v^a)"]
    ZO --> ISo["IS (obj)\nInfoNCE on g_o(z_v^o)"]
    ZA --> VTAa["VTA (attr)\n1-cos(z_v^a,h_a)"]
    ZO --> VTAo["VTA (obj)\n1-cos(z_v^o,h_o)"]
  end

  %% =======================
  %% Cross-modal alignment
  %% =======================
  subgraph Align["(3) Cross-Modal Multi-Level Alignment"]
    ZA --- AL1["L_attr = 1-cos(z_v^a,h_a)"] --- HA
    ZO --- AL2["L_obj = 1-cos(z_v^o,h_o)"] --- HO
    ZP --- AL3["L_pair = InfoNCE\n(z_v^p vs z_g(a,o))"] --- ZG
  end

  %% =======================
  %% Training & inference
  %% =======================
  Align --> Ltotal["Training: L_total = L_graph + L_vsem + L_align"]
  ZP --> Infer["Inference over C_test:\nargmax_{(a,o)} [s_a+s_o+s_comp]"]
  ZG --> Infer
  ZA --> Infer
  ZO --> Infer
