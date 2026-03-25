from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
import math
from functools import lru_cache

app = Flask(__name__)

# --- FUNZIONI DI UTILITÀ E DATI ---
@lru_cache(maxsize=1)
def ottieni_dati_btp_veloci():
    try:
        dati_btp = yf.download("IITB.MI", period="5y", progress=False)
        if not dati_btp.empty:
            returns = dati_btp['Close'].pct_change().dropna()
            if isinstance(returns, pd.DataFrame): 
                returns = returns.iloc[:, 0]
            return float(returns.mean() * 252), float(returns.std() * np.sqrt(252))
    except Exception:
        pass 
    return 0.035, 0.04 

def aggiungi_tasse_ombra(g, tasse_tot):
    rend_pos = [r for r in g['rend_netto_pre_tax'] if r > 0]
    somma_pos = sum(rend_pos) if rend_pos else 0
    g['tasse'], g['rend_reale'] = [], []
    for rn, b, inf in zip(g['rend_netto_pre_tax'], g['bollo'], g['inflazione']):
        cgt = tasse_tot * (rn/somma_pos) if rn > 0 and somma_pos > 0 else 0
        g['tasse'].append(cgt + b)
        g['rend_reale'].append(rn - (cgt + b) - inf)

# --- LOGICHE DEI PRODOTTI ---
def simula_generazione(cap, anni, scenario, uscita, scelta_percorso, ha_bonus, perc_gesav, perc_fi, perc_oicr, usa_strategia):
    a_gs, a_fi, a_oicr, a_inf = scenario['a_gs'], scenario['a_fi'], scenario['a_oicr'], scenario['a_inf']
    COSTO_IN = 10.0; TRATT_GS = 0.021; C_FI = 0.020; C_OICR = 0.021; BOLLO = 0.002
    
    premio_netto = cap - COSTO_IN
    comm = COSTO_IN
    cap_gs = premio_netto * perc_gesav
    cap_f = premio_netto * (perc_fi + perc_oicr)

    if usa_strategia:
        val_gs = cap_gs + (cap_f * 0.90)
        val_fi = (premio_netto * perc_fi) * 0.10
        val_oicr = (premio_netto * perc_oicr) * 0.10
        trav_fi = (premio_netto * perc_fi * 0.90) / 3
        trav_oicr = (premio_netto * perc_oicr * 0.90) / 3
    else:
        val_gs, val_fi, val_oicr = cap_gs, premio_netto * perc_fi, premio_netto * perc_oicr

    bollo_acc = 0.0
    g = {'anni':[], 'rend_lordo':[], 'costi':[], 'bollo':[], 'inflazione':[], 'rend_netto_pre_tax':[]}
    val_inf = cap; inf_cum = 1.0

    for idx in range(anni):
        inf_cum *= (1 + a_inf[idx])
        v_gs_m, v_fi_m, v_oicr_m = val_gs * (1 + a_gs[idx]), val_fi * (1 + a_fi[idx]), val_oicr * (1 + a_oicr[idx])
        rl_anno = (v_gs_m - val_gs) + (v_fi_m - val_fi) + (v_oicr_m - val_oicr)

        v_gs_n = val_gs * (1 + (a_gs[idx] - TRATT_GS))
        if not usa_strategia or idx > 2: v_gs_n = max(cap_gs, v_gs_n)
        c_gs = max(0, v_gs_m - v_gs_n); comm += c_gs; val_gs = v_gs_n

        v_fi_n = v_fi_m * (1 - C_FI); c_fi = v_fi_m - v_fi_n; comm += c_fi; val_fi = v_fi_n
        v_oicr_n = v_oicr_m * ((1 - C_OICR/3)**3); c_oicr = v_oicr_m - v_oicr_n; comm += c_oicr; val_oicr = v_oicr_n

        if usa_strategia and idx <= 2:
            val_gs -= (trav_fi + trav_oicr); val_fi += trav_fi; val_oicr += trav_oicr
        
        c_tot = c_gs + c_fi + c_oicr + (COSTO_IN if idx == 0 else 0)
        b_anno = (val_fi + val_oicr) * BOLLO; bollo_acc += b_anno
        
        g['anni'].append(str(idx+1)); g['rend_lordo'].append(rl_anno); g['costi'].append(c_tot)
        g['bollo'].append(b_anno); g['inflazione'].append(val_inf * a_inf[idx])
        g['rend_netto_pre_tax'].append(rl_anno - c_tot)
        val_inf = val_gs + val_fi + val_oicr

    v_lordo = val_gs + val_fi + val_oicr
    imp_pen, mag_m, b_fed, pen_perc = 0, 0, 0, 0
    if uscita == 'r':
        pen_perc = {1:1.0, 2:0.025, 3:0.02, 4:0.015, 5:0.01, 6:0.005}.get(anni, 0.0)
        imp_pen = v_lordo * pen_perc; comm += imp_pen; g['costi'][-1] += imp_pen; g['rend_netto_pre_tax'][-1] -= imp_pen
    elif uscita == 'd':
        mag_m = (val_fi + val_oicr) * 0.002; g['rend_lordo'][-1] += mag_m; g['rend_netto_pre_tax'][-1] += mag_m

    v_scontato = v_lordo - imp_pen
    if ha_bonus and anni >= 12:
        b_fed = v_scontato * 0.02; comm -= b_fed; g['costi'][-1] -= b_fed; g['rend_netto_pre_tax'][-1] += b_fed

    v_cli = v_scontato + mag_m + b_fed
    peso_gs, peso_f = val_gs/v_lordo if v_lordo > 0 else 0, (val_fi+val_oicr)/v_lordo if v_lordo > 0 else 0
    plus_gs, plus_f = max(0, v_cli*peso_gs - cap_gs), max(0, v_cli*peso_f - cap_f)
    plus_tot = plus_gs + plus_f - COSTO_IN
    tasse = (plus_gs * 0.18) + (plus_f * 0.26)
    v_netto = v_cli - tasse - bollo_acc
    v_reale = v_netto / inf_cum

    return cap, v_netto, v_reale, sum(g['rend_lordo']), comm, tasse, bollo_acc, v_netto - v_reale, g, "GenerAzione", "€"

def simula_rinnova(cap, anni, scenario, uscita, opz, promo, cambio):
    simb = "$" if opz == 3 else "€"
    if (opz == 2 and cap > 600000) or (opz != 2 and cap > 1000000):
        cap = 600000 if opz == 2 else 1000000

    costi_in = 60.0
    if opz == 3: costi_in *= cambio

    a_gs, a_inf = scenario['a_gs'], scenario['a_inf']
    TRATT_1 = 0.0070 if promo else 0.0110
    TRATT_STD = 0.0110

    val = cap - costi_in
    comm = costi_in
    g = {'anni':[], 'rend_lordo':[], 'costi':[], 'bollo':[], 'inflazione':[], 'rend_netto_pre_tax':[]}
    val_inf = cap; inf_cum = 1.0

    for idx in range(anni):
        inf_cum *= (1 + a_inf[idx])
        tratt = TRATT_1 if idx == 0 else TRATT_STD
        v_m = val * (1 + a_gs[idx]); v_n = val * (1 + (a_gs[idx] - tratt))
        
        rl = v_m - val; c_anno = max(0, v_m - v_n) + (costi_in if idx == 0 else 0)
        comm += max(0, v_m - v_n); val = v_n

        g['anni'].append(str(idx+1)); g['rend_lordo'].append(rl); g['costi'].append(c_anno)
        g['bollo'].append(0.0); g['inflazione'].append(val_inf * a_inf[idx])
        g['rend_netto_pre_tax'].append(rl - c_anno)
        val_inf = val

    imp_pen, b_fed, pen_perc = 0, 0, 0
    if uscita == 'r':
        pen_perc = {1:1.0, 2:0.02, 3:0.015, 4:0.01, 5:0.005}.get(anni, 0.0)
        imp_pen = val * pen_perc; comm += imp_pen; g['costi'][-1] += imp_pen; g['rend_netto_pre_tax'][-1] -= imp_pen
        v_sco = val - imp_pen
        if anni >= 6: b_fed = v_sco * 0.01; comm -= b_fed; g['costi'][-1] -= b_fed; g['rend_netto_pre_tax'][-1] += b_fed
        v_cli = v_sco + b_fed
    elif uscita == 'd':
        if val < (cap - costi_in):
            diff = (cap - costi_in) - val; v_cli = cap - costi_in; comm -= diff; g['costi'][-1] -= diff; g['rend_netto_pre_tax'][-1] += diff
        else:
            if anni >= 6: b_fed = val * 0.01; comm -= b_fed; g['costi'][-1] -= b_fed; g['rend_netto_pre_tax'][-1] += b_fed
            v_cli = val + b_fed

    plus_tot = max(0, v_cli - cap)
    tasse = plus_tot * 0.19
    v_netto = v_cli - tasse
    v_reale = v_netto / inf_cum

    return cap, v_netto, v_reale, sum(g['rend_lordo']), sum(g['costi']), tasse, 0.0, v_netto - v_reale, g, "Rinnova Valore Bonus", simb

def simula_sviluppo(cap, anni, scenario, uscita, is_piu, usa_bilancia):
    if cap > 500000: cap = 500000
    a_gs, a_fi, a_inf = scenario['a_gs'], scenario['a_fi'], scenario['a_inf']

    DIRITTI = 5.0 if is_piu else 10.0
    prem_inv = cap - 150.0 - DIRITTI
    comm = 150.0 + DIRITTI
    
    if usa_bilancia: val_gs, val_fi, trav_bim = prem_inv * 0.90, prem_inv * 0.10, (prem_inv * 0.50)/12
    else: val_gs, val_fi, trav_bim = prem_inv * 0.40, prem_inv * 0.60, 0.0
    fisc_gs, fisc_fi, gar_gs = val_gs, val_fi, val_gs

    bollo_acc = 0.0
    g = {'anni':[], 'rend_lordo':[], 'costi':[], 'bollo':[], 'inflazione':[], 'rend_netto_pre_tax':[]}
    val_inf = cap; inf_cum = 1.0

    for idx in range(anni):
        inf_cum *= (1 + a_inf[idx])
        c_pre = comm; v_start = val_gs + val_fi
        rl_gs = a_gs[idx]; tratt = 0.0175 + (math.floor((rl_gs - 0.04)/0.001)*0.0003 if rl_gs > 0.04 else 0)
        rn_gs = rl_gs - tratt

        if usa_bilancia and idx <= 1:
            tb_lg, tb_ng = (1+rl_gs)**(1/6)-1, (1+rn_gs)**(1/6)-1
            tb_lf, tb_nf = (1+a_fi[idx])**(1/6)-1, (1+a_fi[idx]-0.0185)**(1/6)-1
            for _ in range(6):
                vt_g, vt_f = val_gs*(1+tb_lg), val_fi*(1+tb_lf)
                val_gs *= (1+tb_ng); val_fi *= (1+tb_nf)
                comm += max(0, vt_g-val_gs) + max(0, vt_f-val_fi)
                val_gs -= trav_bim; val_fi += trav_bim; fisc_gs -= trav_bim; fisc_fi += trav_bim; gar_gs -= trav_bim
        else:
            vm_g = val_gs * (1+rl_gs); vn_g = val_gs * (1+rn_gs); comm += max(0, vm_g-vn_g); val_gs = vn_g
            vm_f = val_fi * (1+a_fi[idx]); vn_f = vm_f * (1-0.0185); comm += (vm_f-vn_f); val_fi = vn_f
            
        b_anno = val_fi * 0.002; bollo_acc += b_anno
        c_anno = comm - c_pre + (150+DIRITTI if idx==0 else 0)
        rl_anno = (val_gs+val_fi) - v_start + (comm - c_pre)
        
        g['anni'].append(str(idx+1)); g['rend_lordo'].append(rl_anno); g['costi'].append(c_anno)
        g['bollo'].append(b_anno); g['inflazione'].append(val_inf * a_inf[idx])
        g['rend_netto_pre_tax'].append(rl_anno - c_anno)
        val_inf = val_gs + val_fi

    imp_pen, mag_m, pen_perc = 0, 0, 0
    if uscita == 'r':
        pen_perc = {1:1.0, 2:0.02, 3:0.015, 4:0.01, 5:0.005}.get(anni, 0.0)
        imp_pen = val_gs * pen_perc; comm += imp_pen + 10.0; g['costi'][-1] += imp_pen + 10.0; g['rend_netto_pre_tax'][-1] -= (imp_pen+10)
        v_cli_g, v_cli_f = val_gs - imp_pen, val_fi
        v_tot = v_cli_g + v_cli_f - 10.0
    else:
        v_cli_g = max(val_gs, gar_gs)
        if val_gs < gar_gs: diff=gar_gs-val_gs; comm-=diff; g['costi'][-1]-=diff; g['rend_netto_pre_tax'][-1]+=diff
        mag_m = val_fi * 0.002; v_cli_f = val_fi + mag_m; comm -= mag_m; g['rend_lordo'][-1]+=mag_m; g['rend_netto_pre_tax'][-1]+=mag_m
        v_tot = v_cli_g + v_cli_f

    plus_gs, plus_fi = max(0, v_cli_g - fisc_gs), max(0, v_cli_f - fisc_fi)
    plus_tot = plus_gs + plus_fi
    tasse = (plus_gs * 0.19) + (plus_fi * 0.26)
    v_netto = v_tot - tasse - bollo_acc
    v_reale = v_netto / inf_cum

    return cap, v_netto, v_reale, sum(g['rend_lordo']), sum(g['costi']), tasse, bollo_acc, v_netto - v_reale, g, "GeneraSviluppo Sost.", "€"

def simula_valore(cap, anni, scenario, uscita, is_piu, traguardo, t_ob):
    if cap > 500000: cap = 500000
    a_gs, a_oicr, a_inf = scenario['a_gs'], scenario['a_oicr'], scenario['a_inf']

    DIRITTI = 5.0 if is_piu else 10.0
    prem_inv = cap - DIRITTI
    cap_prot = prem_inv
    val_gs = cap_prot / ((1+t_ob)**traguardo); val_oicr = prem_inv - val_gs
    fisc_gs, fisc_oicr, stor_gs = val_gs, val_oicr, val_gs

    comm = DIRITTI; bollo_acc = 0.0; inf_cum = 1.0
    g = {'anni':[], 'rend_lordo':[], 'costi':[], 'bollo':[], 'inflazione':[], 'rend_netto_pre_tax':[]}
    val_inf = cap

    for mese in range(1, anni*12 + 1):
        if mese % 12 == 1 or mese == 1:
            v_start_anno = val_gs + val_oicr; c_pre = comm; b_anno = 0.0
        
        idx = math.ceil(mese/12) - 1
        tm_g = (1+a_gs[idx])**(1/12)-1; tm_o = (1+a_oicr[idx])**(1/12)-1; tm_i = (1+a_inf[idx])**(1/12)-1
        inf_cum *= (1+tm_i)
        
        val_gs *= (1+tm_g); val_oicr *= (1+tm_o)
        
        if mese % 12 == 0: b_anno = val_oicr * 0.002; bollo_acc += b_anno

        if mese == 8 or (mese > 8 and (mese-8)%4 == 0):
            a_pol = math.ceil(mese/12)
            tg = 0.021 if a_pol<=5 else (0.019 if a_pol<=10 else (0.017 if a_pol<=15 else 0.015))
            c_quad = (val_gs+val_oicr) * (tg/3); c_eff = min(c_quad, val_oicr)
            val_oicr -= c_eff; comm += c_eff
            vc_post = val_gs + val_oicr
            
            if vc_post >= (cap_prot * 1.10): cap_prot = vc_post
            
            anni_rim = max(0.0, traguardo - (mese/12))
            targ_gs = cap_prot / ((1+t_ob)**anni_rim)
            
            sposta = val_oicr if vc_post < targ_gs else targ_gs - val_gs
            val_gs += sposta; val_oicr -= sposta
            
            if sposta > 0: fisc_gs += sposta; stor_gs += sposta
            elif sposta < 0: 
                q = abs(sposta)/(val_gs - sposta); fisc_gs *= (1-q); stor_gs *= (1-q)
            fisc_oicr -= sposta

        if mese % 12 == 0:
            c_anno = comm - c_pre + (DIRITTI if idx==0 else 0)
            rl_anno = (val_gs+val_oicr) - v_start_anno + (comm - c_pre)
            g['anni'].append(str(idx+1)); g['rend_lordo'].append(rl_anno); g['costi'].append(c_anno)
            g['bollo'].append(b_anno); g['inflazione'].append(val_inf * a_inf[idx])
            g['rend_netto_pre_tax'].append(rl_anno - c_anno)
            val_inf = val_gs + val_oicr

    v_tot = val_gs + val_oicr
    imp_pen, mag_m = 0, 0
    if uscita == 'r':
        if anni <= 6:
            imp_pen = v_tot * 0.02; comm += imp_pen; g['costi'][-1] += imp_pen; g['rend_netto_pre_tax'][-1] -= imp_pen; v_tot -= imp_pen
        if anni >= traguardo and v_tot < stor_gs:
            diff = stor_gs - v_tot; comm -= diff; v_tot = stor_gs; g['costi'][-1] -= diff; g['rend_netto_pre_tax'][-1] += diff
    else:
        mag_m = val_oicr * 0.002; comm -= mag_m; g['rend_lordo'][-1] += mag_m; g['rend_netto_pre_tax'][-1] += mag_m; v_tot += mag_m
        if v_tot < stor_gs:
            diff = stor_gs - v_tot; comm -= diff; v_tot = stor_gs; g['costi'][-1] -= diff; g['rend_netto_pre_tax'][-1] += diff

    p_gs = (val_gs/(val_gs+val_oicr)) if (val_gs+val_oicr)>0 else 0
    p_oi = (val_oicr/(val_gs+val_oicr)) if (val_gs+val_oicr)>0 else 0
    plus_tot = (max(0, v_tot*p_gs - fisc_gs)) + (max(0, v_tot*p_oi - fisc_oicr))
    tasse = (max(0, v_tot*p_gs - fisc_gs) * 0.19) + (max(0, v_tot*p_oi - fisc_oicr) * 0.26)
    v_netto = v_tot - tasse - bollo_acc
    v_reale = v_netto / inf_cum

    return cap, v_netto, v_reale, sum(g['rend_lordo']), sum(g['costi']), tasse, bollo_acc, v_netto - v_reale, g, "Valore Futuro", "€"


# --- ROUTE PRINCIPALI ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simula', methods=['POST'])
def api_simula():
    d = request.json
    
    # 1. PARAMETRI BASE CONDIVISI
    capitale = float(d.get('capitale', 50000))
    anni = int(d.get('anni', 10))
    uscita = d.get('uscita', 'r')
    tassi = d.get('tassi', 'f')

    # 2. GENERAZIONE SCENARIO UNICO PER TUTTI I PRODOTTI
    if tassi == 'f':
        r_gs = float(d.get('tassi_gs', 3.5)) / 100
        r_fi = float(d.get('tassi_fi', 5.0)) / 100
        r_oicr = float(d.get('tassi_oicr', 4.0)) / 100
        inf = float(d.get('tassi_inf', 2.0)) / 100
        a_gs, a_fi, a_oicr, a_inf = np.full(anni, r_gs), np.full(anni, r_fi), np.full(anni, r_oicr), np.full(anni, inf)
    else:
        rm_obbl, vol_obbl = ottieni_dati_btp_veloci()
        a_gs = np.clip(np.random.normal(rm_obbl, vol_obbl, anni), 0.004, 0.075)
        a_fi = np.random.normal(0.085, 0.152, anni)
        a_oicr = np.random.normal(0.075, 0.140, anni)
        a_inf = np.clip(np.random.normal(0.020, 0.012, anni), 0.0, 0.05)

    scenario = {'tipo': tassi, 'a_gs': a_gs, 'a_fi': a_fi, 'a_oicr': a_oicr, 'a_inf': a_inf}

    # 3. MOTORE DI ESECUZIONE DINAMICO PER SINGOLO PRODOTTO
    def esegui_prodotto(p):
        prodotto = p.get('prodotto', 'GenerAzione')
        
        try:
            if prodotto == 'GenerAzione':
                res = simula_generazione(capitale, anni, scenario, uscita,
                    int(p.get('ga_percorso', 1)), bool(p.get('ga_bonus', False)),
                    float(p.get('ga_gesav', 40)) / 100, float(p.get('ga_fi', 40)) / 100,
                    float(p.get('ga_oicr', 20)) / 100, bool(p.get('ga_strategia', True))
                )
            elif prodotto == 'Rinnova Valore Bonus':
                res = simula_rinnova(capitale, anni, scenario, uscita,
                    int(p.get('rv_opzione', 1)), bool(p.get('rv_promo', True)),
                    float(p.get('rv_cambio', 1.08))
                )
            elif prodotto == 'GeneraSviluppo':
                res = simula_sviluppo(capitale, anni, scenario, uscita,
                    bool(p.get('gs_piu', False)), bool(p.get('gs_bilancia', True))
                )
            elif prodotto == 'Valore Futuro':
                res = simula_valore(capitale, anni, scenario, uscita,
                    bool(p.get('vf_piu', False)), int(p.get('vf_traguardo', 15)),
                    float(p.get('vf_tasso', 3.0)) / 100
                )
        except Exception as e:
            raise e
            
        c_iniz, v_netto, v_reale, sum_rl, costi_tot, tasse_tot, bollo_acc, erosione, g, nome, simb = res
        aggiungi_tasse_ombra(g, tasse_tot + bollo_acc)
        
        return {
            'nome': nome, 'simbolo': simb,
            'kpi': {
                'cap_iniziale': c_iniz, 'netto': v_netto, 'reale': v_reale,
                'cagr_netto': (((v_netto/c_iniz)**(1/anni))-1) * 100 if c_iniz > 0 else 0,
                'cagr_reale': (((v_reale/c_iniz)**(1/anni))-1) * 100 if c_iniz > 0 else 0
            },
            'grafico': {
                'anni': g['anni'], 'rend_reale': g['rend_reale'], 
                'inflazione': g['inflazione'], 'costi': g['costi'], 'tasse': g['tasse'] 
            },
            'confronto': {
                'cap_iniziale': c_iniz, 'r_lordo_v': sum_rl,
                'r_netto_v': v_netto - c_iniz, 'r_reale_v': v_reale - c_iniz,
                'costi_tot': sum(g['costi']), 'tasse_tot': tasse_tot + bollo_acc, 'erosione': erosione
            }
        }

    # 4. CICLO SU TUTTI I PRODOTTI INVIATI DAL FRONTEND
    lista_prodotti = d.get('prodotti', [])
    risultati_finali = []
    
    for p in lista_prodotti:
        risultati_finali.append(esegui_prodotto(p))

    # Info tassi (uguale per tutti)
    tab_dati = [{'anno': i + 1, 'gs': float(a_gs[i]), 'fi': float(a_fi[i]), 'oicr': float(a_oicr[i]), 'inf': float(a_inf[i])} for i in range(anni)]
    tassi_info = {
        'tipo': tassi, 'dettaglio': tab_dati,
        'medie': {
            'gs': float((np.prod(a_gs+1))**(1/anni)-1) if anni>0 else 0,
            'fi': float((np.prod(a_fi+1))**(1/anni)-1) if anni>0 else 0,
            'oicr': float((np.prod(a_oicr+1))**(1/anni)-1) if anni>0 else 0,
            'inf': float((np.prod(a_inf+1))**(1/anni)-1) if anni>0 else 0
        }
    }

    return jsonify({
        'tassi_info': tassi_info,
        'risultati': risultati_finali
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)