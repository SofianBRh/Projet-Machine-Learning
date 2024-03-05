OUVRIR UN TERMINAL DANS LE DOSSIER PROJET 

python -m venv --help
pour voir la courte docu de venv

AFFICHER LA VERSION
python --version



// WINDOWS //

CREER UN ENVIRONNEMENT
python -m venv <NOM>

ACTIVER L ENVIRONNEMENT
Activer env
env/Scripts/activate

GENERER LE FICHIER REQUIREMENTS
pip freeze > requirements.txt
pour créer la liste des requis

INSTALLER LE FICHIER REQUIREMENTS
pip install -r requirements.txt
pour installer le contenu de requirements.txt

AFFICHER LE CONTENU DE REQUIREMENT
cat/type requirements.txt
pour afficher le content

DESACTIVER L ENVIRONNEMENT
deactivate
pour quitter le venv / switcher dans le cas où on est déjà dans un autre venv


// MAC //


CREER ENVIRONNEMENT VIRTUEL
python -m venv <nom> dans le terminal

ACTIVER L ENVIRONNEMENT
Source <nom>/bin/activate

GENERER LE FICHIER REQUIREMENTS
pip freeze > requirements.txt
pour créer la liste des requis

INSTALLER LE FICHIER REQUIREMENTS
pip install -r requirements.txt
pour installer le contenu de requirements.txt

AFFICHER LE CONTENU DE REQUIREMENT
cat/type requirements.txt
pour afficher le content

DESACTIVER L ENVIRONNEMENT
deactivate
pour quitter le venv / switcher dans le cas où on est déjà dans un autre venv
