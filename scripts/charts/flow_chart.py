from pathlib import Path
from graphviz import Digraph

def create_pipeline_diagram():
    dot = Digraph(comment='ML Pipeline', format='pdf') 
    
    dot.attr(rankdir='LR', size='10', dpi='300')
    
    # Standard-Node-Style
    dot.attr('node', shape='box', style='filled', fillcolor='white', fontname='Helvetica')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # 1. DATENQUELLE
    dot.node('Raw', 'Rohdaten\n(Airbnb CSV)', shape='cylinder', fillcolor='#E6E6E6')
    
    # 2. SPLIT 
    dot.node('Split', 'Train-Test Split\n(Random State = 42)', shape='diamond', fillcolor='#FFF2CC')
    
    # TRAINING
    with dot.subgraph(name='cluster_train') as c:
        c.attr(label='Training Phase (Lernen)', style='dashed', color='grey')
        c.node('TrainData', 'Trainingsdaten\n(80%)')
        c.node('Fit', 'Pipeline .fit()\n(Lerne Median,\nMean, Skala)', color='blue', fontcolor='blue')
        c.node('TransTrain', 'Pipeline .transform()\n(Wende an)')
        c.node('ModelFit', 'Modell Training\n(Fit auf transformierten Daten)', shape='component', fillcolor='#D5E8D4')
        
        # Kanten im Training Cluster
        c.edge('TrainData', 'Fit')
        c.edge('Fit', 'TransTrain', label='Parameter')
        c.edge('TransTrain', 'ModelFit')

    # 4. TEST
    with dot.subgraph(name='cluster_test') as c:
        c.attr(label='Inferenz Phase (Evaluation)', style='dashed', color='grey')
        c.node('TestData', 'Testdaten\n(20%)')
        c.node('TransTest', 'Pipeline .transform()\n(Benutze gelernte Parameter)', color='red', fontcolor='red')
        c.node('Predict', 'Vorhersage\n(Prediction)', shape='component', fillcolor='#D5E8D4')
        c.node('Eval', 'Evaluation\n(RMSE, MAE, R²)', shape='note')

        # Kanten im Test Cluster
        c.edge('TestData', 'TransTest')
        c.edge('TransTest', 'Predict')
        c.edge('Predict', 'Eval')

    # VERBINDUNGEN
    dot.edge('Raw', 'Split')
    dot.edge('Split', 'TrainData')
    dot.edge('Split', 'TestData')
    
    # Verbindung Fit -> Transform Test
    dot.edge('Fit', 'TransTest', label='Übertrage\nRegeln', style='dotted', constraint='false', color='blue')
    
    # Modell -> Prediction
    dot.edge('ModelFit', 'Predict', label='Gelerntes\nModell', style='bold')

    # Ausgabeve
    output_dir = Path(__file__).resolve().parents[1] / "charts/images"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "pipeline_flowchart_landscape"
    dot.render(output_path, view=False, cleanup=True)

    print(f"Diagramm erstellt: {output_path}.pdf")

if __name__ == '__main__':
    create_pipeline_diagram()