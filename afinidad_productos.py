import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from collections import Counter
import os
from typing import List, Tuple
from tqdm import tqdm
from community import community_louvain
from plot_graph import *
from netgraph import Graph

class AfinidadProductos:
    def __init__(self, filename : str, col_nivel_analisis : str) -> None:
        self.filename = filename 
        self.col_nivel_analisis = col_nivel_analisis
    
    def _leer_todos_los_archivos(self) -> pd.DataFrame:
        """
        Este método lee todos los archivos de la carpeta data y los concatena en un solo dataframe
        """
        df = pd.DataFrame()
        for file in [f for f in os.listdir('data') if (f.endswith('.csv') or f.endswith('.xlsx'))]:
            if file.endswith('.csv'):
                df = pd.concat([df, pd.read_csv(f'data\{file}')])
            elif file.endswith('.xlsx'):
                df = pd.concat([df, pd.read_excel(f'data\{file}')])
        return df

    def cargar_y_procesar_datos(self) -> pd.DataFrame:
        """
        Este método lee la tabla de datos y la devuelve como un DataFrame de pandas después de construir el id de cada fila
        """
        if self.filename.endswith('.csv'):
            print("Leyendo el archivo csv...")
            df = pd.read_csv(f'data\{self.filename}')

        elif self.filename.endswith('.xlsx'):
            print("Leyendo el archivo xlsx...")
            df = pd.read_excel(f'data\{self.filename}')

        elif self.filename == "_all_folder_":
            print("Leyendo todos los archivos de la carpeta...")
            df = self._leer_todos_los_archivos()

        #df.dropna(inplace = True)
        df = df.reset_index(drop=True)
        df = self._crear_id(df)
        self.df = df
        print(f"Se leyó correctamente la información. Un total de {df.shape[0]} registros.")
        return df

    def _crear_id(self, df) -> pd.DataFrame:
        """
        Este método crea un nuevo id en el dataframe que junte transacción y cédula
        """
        df['id'] = df['Transaccion'].astype(str) + '_' + df['Cedula'].astype(str)
        return df

    def _crear_df_transacciones_productos(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Este método crea una tabla pivote que relaciona cada transacción con los productos que contiene
        """
        if self.col_nivel_analisis not in df.columns:
            raise ValueError(f'La columna {self.col_nivel_analisis} no existe en el dataframe')
        else:
            productos = df[['id', self.col_nivel_analisis]].drop_duplicates()
            productos.sort_values(by=['id', self.col_nivel_analisis], inplace=True)
        return productos

    def crear_relaciones_grafo(self, df : pd.DataFrame, min_brecha : float = 0.135) -> List:
        productos = self._crear_df_transacciones_productos(df)

        # Dataframe con las relaciones de slos nodos
        df_nodos = productos.groupby("id")[self.col_nivel_analisis].apply(lambda x : list(itertools.combinations(x, 2))).to_frame(name = 'nodos').reset_index()

        # Nodos totales para el grafo
        _nodos_totales = [item for sublist in list(df_nodos['nodos']) for item in sublist]

        """
        Se seleccionan los nodos de la siguiente forma:
        *) De todas las apariciones de un producto, se dice que está relacionado 
            con otro si al menos aparecen juntos el min_brecha% de las veces.
        """

        # Esto crea las aristas según (x->y) o (y->x)
        nodos_totales = []
        
        print("Comienza la generación del grafo:")
        for categoria in tqdm(productos[self.col_nivel_analisis].unique()):
            relaciones = [x for x in _nodos_totales if x[0] == categoria or x[1] == categoria]
            total_apariciones = productos[productos[self.col_nivel_analisis] == categoria].shape[0]
            contador = Counter(relaciones)

            for k, v in contador.items():
                contador[k] = v / total_apariciones
                if v/total_apariciones > min_brecha:
                    nodos_totales.append(k)

        # Creación de las aristas según el segundo método (x->y), (y->x)
        contador_nodos = Counter(nodos_totales)
        nodos_finales = []

        for indice, pareja in enumerate(contador_nodos):
            if list(contador_nodos.values())[indice] != 1:
                nodos_finales.append(pareja)
        return nodos_finales

    def crear_grafo(self, nodos : List) -> nx.Graph:
        G0 = nx.Graph()
        G0.add_edges_from(nodos)
        
        Gc = sorted(nx.connected_components(G0), key=len, reverse=True)
        G = G0.subgraph(Gc[0])
        print("El grafo se generó correctamente :)")
        return G

    def generar_comunidades(self, G : nx.Graph) -> Tuple:
        partition = community_louvain.best_partition(G)
        unique_coms = np.unique(list(partition.values()))
        return partition, unique_coms 

    def obtener_tabla_comunidades_productos(self, comms : Tuple, exportar = True) -> pd.DataFrame:
        df_comunidades = pd.DataFrame(data = list(comms.items()), columns = [self.col_nivel_analisis, 'comunidad'])
        tabla_comunidades = pd.merge(self.df, df_comunidades, on = self.col_nivel_analisis, how = 'left')

        if exportar:
            print("Exportando tabla...")
            tabla_comunidades.to_excel(f'data\comunidades_{self.filename if filename != "_all_folder_" else "archivos_consolidados.xlsx"}', index = False)
            print("Tabla exportada!")

        return tabla_comunidades

    def _crear_cmap(self, comms : Tuple) -> Tuple:
        colores = ['tab:blue', 'tab:orange',
            'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray',
            'tab:olive', 'tab:cyan']

        cmap = dict(
            zip(
                list(range(len(comms))),
                colores
            )
        )
        return cmap

    def atributos_grafico(self, df : pd.DataFrame, comms : Tuple) -> Tuple:
        cmap_comunidades = self._crear_cmap(comms)
        pos = community_layout(G, comms)
        conteo_categorias = df.groupby(self.col_nivel_analisis)['id'].count().sort_values(ascending = False)
        node_sizes = [conteo_categorias[node] for node in G.nodes()]
        node_color = {node: cmap_comunidades[community_id] for node, community_id in comms.items()}
        return pos, node_sizes, node_color


class GraficoRedComerEsPrimero:
    def __init__(self, G : nx.Graph, node_sizes : List, node_color : List, 
    comms : Tuple, titulo : str = 'Red de Comer es Primero'):
        self.G = G
        self.node_sizes = node_sizes
        self.node_colors = node_color
        self.comms = comms
        self.titulo = titulo

    def generar_grafica(self, guardar = False):
        # Gráfico de la red
        f = plt.figure(figsize=(20, 20))
        max_node_size = .15 * max(self.node_sizes)

        Graph(G,
            node_edge_width=0.2, 
            node_color = self.node_colors,
            node_alpha = 0.6,
            edge_width = 0.25,
            edge_alpha=0.13,
            node_layout='community', 
            node_layout_kwargs=dict(node_to_community=self.comms),
            edge_layout='straight', 
            node_size = dict(zip(G.nodes(), 2 + np.array(self.node_sizes) / max_node_size)),
            node_labels = {node : node.capitalize().replace('y tostados', '').replace('autoservicio', '').replace('comestibles', '').replace('no refrig', '').replace('de gallina', '') for node in G.nodes() if G.degree(node) >= 1},
            node_label_fontdict=dict(size=16, fontfamily='Arial'),
        )

        plt.title(self.titulo, fontsize = 35)
        plt.box(False)

        if guardar:
            f.savefig("productos_complementarios_nov_n.pdf")

        plt.show()

if __name__ == "__main__":
    filename = '_all_folder_'
    columna_categoria = 'Nombre Categoria'

    # Instanciar la clase AfinidadProductos
    productos_comeresprimero = AfinidadProductos(filename, columna_categoria)

    # Obtener el dataframe con los datos
    df = productos_comeresprimero.cargar_y_procesar_datos()

    # Generar los nodos y las relaciones entre los productos
    nodos = productos_comeresprimero.crear_relaciones_grafo(df)
    G = productos_comeresprimero.crear_grafo(nodos)

    # Obtener las comunidades del grafo por el método de Louvain
    comms, unique_coms = productos_comeresprimero.generar_comunidades(G)

    # Obtener los atributos del gráfico
    pos, node_sizes, node_colors = productos_comeresprimero.atributos_grafico(df, comms)

    # Obtener y guardar tabla base con nombre de comunidades de productos
    tabla_comunidades = productos_comeresprimero.obtener_tabla_comunidades_productos(comms, exportar = False)

    # Generar el gráfico
    titulo_grafico = "Red de productos de Comer es Primero \n Octubre"
    grafico = GraficoRedComerEsPrimero(G, node_sizes, node_colors, comms, titulo_grafico)
    grafico.generar_grafica(guardar = True)
