from graphviz import Digraph
from pathlib import Path

def create_validation_diagram():
    """
    Generiert ein Flowchart zur Visualisierung der hybriden Validierungsstrategie.
    Unterscheidet zwischen k-Fold Cross-Validation (klassisch) und 
    Hold-Out mit Early Stopping (modern).
    """
    
    # Initialisierung des Graphen
    dot = Digraph(comment='Hybrid Validation Schema', format='pdf')
    dot.attr(rankdir='LR', size='12', dpi='300', compound='true')
    
    # Globale Knoteneinstellungen
    dot.attr('node', shape='box', style='filled', fillcolor='white', 
             fontname='Helvetica', fontsize='11')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # Eingangsdaten
    dot.node('Input', 'Trainingsdaten\n(80% Split)', shape='cylinder', fillcolor='#E6E6E6')
    dot.node('Switch', '', shape='point', width='0') # Unsichtbarer Weichen-Knoten
    dot.edge('Input', 'Switch')

    # Zweig A: Klassische Modelle (k-Fold CV)
    with dot.subgraph(name='cluster_classical') as c:
        c.attr(label='Pfad A: Klassische Modelle (z.B. Linear, Random Forest)', 
               style='dashed', color='#4C72B0', fontcolor='#4C72B0')
        c.attr(bgcolor='#F0F5F9')

        # Datensplit
        c.node('kFold', 'k-Fold Split\n(k=3)', shape='parallelogram', fillcolor='#FFF2CC')
        
        # Symbolische Darstellung der Iterationsschleife
        with c.subgraph(name='cluster_cv_loop') as loop:
            loop.attr(label='Iterative CV-Schleife (pro Fold)', style='solid', color='grey', bgcolor='white')
            
            loop.node('TrainFold', 'Train Folds\n(k-1)', shape='note', fillcolor='#D5E8D4')
            loop.node('ValFold', 'Val Fold\n(1)', shape='note', fillcolor='#F8CECC')
            
            # Pipeline-Logik zur Vermeidung von Data Leakage
            loop.node('PipeFitCV', 'Pipeline .fit()\n(Lerne Parameter)', fontcolor='blue', color='blue')
            loop.node('ModelTrainCV', 'Modell Training\n(.fit)', shape='component')
            loop.node('EvalCV', 'Evaluation\n(Score)')

            # Datenfluss innerhalb der Schleife
            loop.edge('TrainFold', 'PipeFitCV')
            loop.edge('PipeFitCV', 'ModelTrainCV', label='Transformierte\nDaten')
            
            # Validierungsdaten werden nur transformiert, nicht zum Fitten genutzt
            loop.edge('ValFold', 'EvalCV', label='Transformiere\n(ohne Fit)', style='dotted')
            loop.edge('ModelTrainCV', 'EvalCV')

        # Aggregation der Ergebnisse
        c.node('CVScore', 'Aggregierte\nMetriken', shape='diamond', fillcolor='#E1D5E7')

        # Verbindungen im Cluster
        c.edge('kFold', 'TrainFold', lhead='cluster_cv_loop')
        c.edge('EvalCV', 'CVScore', ltail='cluster_cv_loop', label='Aggregiere\nk Ergebnisse')


    # Zweig B: Moderne Modelle (Hold-Out & Early Stopping)
    with dot.subgraph(name='cluster_modern') as c:
        c.attr(label='Pfad B: Iterative Modelle (z.B. XGBoost, TabNet)', 
               style='dashed', color='#C44E52', fontcolor='#C44E52')
        c.attr(bgcolor='#FFF0F0')

        # Datensplit
        c.node('HoldOut', 'Hold-Out Split\n(Train / Val)', shape='parallelogram', fillcolor='#FFF2CC')

        # Preprocessing (Strikte Trennung)
        c.node('TrainInt', 'Interner Train', shape='note', fillcolor='#D5E8D4')
        c.node('ValInt', 'Interner Val', shape='note', fillcolor='#F8CECC')

        c.node('PipeFitInt', 'Pipeline .fit()\n(nur auf Train)', fontcolor='blue', color='blue')
        
        # Training mit Validierungsüberwachung
        c.node('ESTrain', 'Training mit\nEarly Stopping', shape='component', fillcolor='#D5E8D4')
        
        # Ergebnis
        c.node('BestModel', 'Optimiertes\nModell', shape='diamond', fillcolor='#E1D5E7')

        # Verbindungen im Cluster
        c.edge('HoldOut', 'TrainInt')
        c.edge('HoldOut', 'ValInt')

        c.edge('TrainInt', 'PipeFitInt')
        c.edge('PipeFitInt', 'ESTrain', label='Transformierte\nDaten')
        
        # Validierungsdaten fließen in das Training zur Überwachung (eval_set)
        c.edge('ValInt', 'ESTrain', label='Überwachung\n(eval_set)', style='dotted', constraint='false')
        
        c.edge('ESTrain', 'BestModel', label='Stopp bei\nKonvergenz')


    # Hauptverbindungen (Entscheidungslogik)
    dot.edge('Switch', 'kFold', label='Nicht-Iterativ', color='#4C72B0', fontcolor='#4C72B0', penwidth='2')
    dot.edge('Switch', 'HoldOut', label='Iterativ', color='#C44E52', fontcolor='#C44E52', penwidth='2')


    # Ausgabe
    output_dir = Path(__file__).resolve().parents[1] / "charts/images"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'validation_schema'
    
    dot.render(output_path, view=False, cleanup=True)
    print(f"Diagramm erfolgreich erstellt: {output_path}.pdf")

if __name__ == '__main__':
    create_validation_diagram()