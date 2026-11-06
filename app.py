import streamlit as st
import pandas as pd
import joblib

# On charge le modele et le scaler entraines (mis en cache pour ne les lire qu'une fois)
@st.cache_resource
def charger():
    modele = joblib.load("modele_logistique.joblib")
    scaler = joblib.load("scaler.joblib")
    return modele, scaler

modele, scaler = charger()

st.title("Détecteur de fraude bancaire")
st.write("Entre une transaction, le modèle estime le risque de fraude.")

# --- Les champs a remplir ---
type_tx = st.selectbox("Type de transaction", ["TRANSFER", "CASH_OUT"])
amount = st.number_input("Montant", min_value=0.0, value=181.0, step=100.0)
oldbalanceOrg = st.number_input("Solde de l'émetteur AVANT", min_value=0.0, value=181.0, step=100.0)
newbalanceOrig = st.number_input("Solde de l'émetteur APRÈS", min_value=0.0, value=0.0, step=100.0)
oldbalanceDest = st.number_input("Solde du destinataire AVANT", min_value=0.0, value=0.0, step=100.0)
newbalanceDest = st.number_input("Solde du destinataire APRÈS", min_value=0.0, value=0.0, step=100.0)

if st.button("Analyser la transaction"):
    # On fabrique les memes indices que pour l'entrainement
    vide_tout = 1 if amount == oldbalanceOrg else 0
    erreur_emetteur = oldbalanceOrg - amount - newbalanceOrig
    erreur_dest = newbalanceDest - oldbalanceDest - amount
    est_transfer = 1 if type_tx == "TRANSFER" else 0

    # Les colonnes DOIVENT etre dans le meme ordre qu'a l'entrainement
    ligne = pd.DataFrame([{
        "step": 1,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "vide_tout": vide_tout,
        "erreur_emetteur": erreur_emetteur,
        "erreur_dest": erreur_dest,
        "est_transfer": est_transfer,
    }])

    # Mise a l'echelle puis prediction de la probabilite de fraude
    proba = modele.predict_proba(scaler.transform(ligne))[0, 1]

    st.write(f"Probabilité de fraude estimée : **{100*proba:.1f} %**")
    if proba > 0.5:
        st.error("Transaction suspecte : probablement une FRAUDE")
    else:
        st.success("Transaction normale")

    st.caption(f"Indice clé — vide tout le compte : {'oui' if vide_tout else 'non'}")
