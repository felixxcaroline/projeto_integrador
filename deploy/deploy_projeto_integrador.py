import streamlit as st
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestRegressor 



st.set_page_config(page_title = 'Simulador - Case Ifood',
                       page_icon = 'logo_dh.jpeg',
                       layout='wide',
                       initial_sidebar_state = 'expanded')

st.title('Simulador - Pré Pagamento de Financiamento')
with st.expander('Descrição do App',expanded=False):
        st.markdown('O objetivo principal desta ferramenta é realizar predições sobre o percentual do contrato em que o cliente irá terminar o financiamento...')
    
st.subheader('Favor preencher todos os campos')
#seleção de features
cat_1 = st.selectbox(
             'Modelo do Veículo',
             ('','Corolla','Corolla Cross','Yaris','Etios','Hilux','SW4','OT - RAV4','OT - Prius','OT - Lexus','OT - Empilhadeira','OT - Camry','OT - Prado'))
cat_2 = st.selectbox(
             'Estado do Veículo',
             ('','Novo','Usado'))
cat_3 = st.selectbox(
             'Quantidade de Parcelas',
             ('',12,24,36,48,60))
cat_4 = st.selectbox(
             'Loja Veículo',
             ('','Grupo','Independente Grupo','Independente'))
cat_5 = st.selectbox(
             'Tipo de Produto',
             ('','CICLO','CDC'))
cat_6 = st.selectbox(
             'Financiamento por TCM',
             ('','SIM','NAO'))
cat_7 = st.selectbox(
             'Região da Compra',
             ('','Sudeste','Nordeste','Sul','Centro-Oeste','Norte'))  
cat_8 = st.number_input('Valor do veículo',format="%.2f",help='Favor informar o valor do veículo, com vírgula')
cat_9 = st.slider('Percentual de Entrada',0,100,step=1)
cat_10 =  st.slider('Idade do veículo (anos)',0,10,step=1)
cat_11 = st.selectbox(
             'Categoria Score',
             ('','A','B','C','D','X'))



Xtest = pd.DataFrame({'DOWN_PAYMENT': [cat_9], 'QTD_PRAZO': [cat_3], 'ESTADO_VEICULO': [cat_2], 'FL_TCM': [cat_6], 
                                      'NS_RANGE_REGIAO': [cat_7], 'NS_RATING_CRIVO': [cat_11], 'GRUPO_LOJAS': [cat_4], 
                                      'VLR_VEIC': [cat_8], 
                                      'TIPO_PROD': [cat_5], 'IDADE_VEIC': [cat_10], 'CAT_VEICULOS': [cat_1]})


tcm = {'Sim':1,'Nao':0}
Xtest['FL_TCM'] = Xtest['FL_TCM'].replace(tcm)

de_para_1  = {'Etios':'C','Corolla':'B','SW4':'A','Hilux':'A','OT - Prius':'A','OT - Camry':'A','OT - RAV4':'A','Outros':'C','OT - Lexus':'A','OT - Empilhadeira':'C','Yaris':'C','OT - Prado':'A','Corolla Cross':'B'}
Xtest['CAT_VEICULOS'] = Xtest['CAT_VEICULOS'].replace(de_para_1)
de_para_2 = {'Novo':'N','Usado':'U'}
Xtest['ESTADO_VEICULO'] = Xtest['CAT_VEICULOS'].replace(de_para_2)

Xtest.info()


local = open('C:\Dev\digital_house\projeto_integrador\deploy\pickle_rfr_tuned_select.pkl', 'rb')
mdl_rfr = pickle.load(local)

cols = mdl_rfr.feature_names_in_.tolist()
Xtest_new = pd.get_dummies(Xtest,drop_first=True)
Xtest_new = Xtest_new.reindex(columns=cols).fillna(0)
ypred = mdl_rfr.predict(Xtest_new)

st.write('O cliente irá termir o contrato de',cat_3, ' parcelas em ',round(ypred[0],2),'% do contrato, aproximadamente na parcela ', round(cat_3*round((ypred[0]/100),2),0))
