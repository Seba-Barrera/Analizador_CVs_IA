############################################################################################
############################################################################################
# App de Analizador de CVs
############################################################################################
############################################################################################


# https://platform.openai.com/account/api-keys
# https://openai.com/pricing


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [A] Importacion de librerias
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# Obtener versiones de paquetes instalados
# !pip list > requirements.txt

import streamlit as st

# librerias para data
import pandas as pd

# libreria para ver imagenes
from PIL import Image

# librerias para graficos
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# librerias de IA
from openai import OpenAI
from pydantic import BaseModel

# libreria para manipular archivos
import zipfile
import fitz  # PyMuPDF
import io



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [B] Creacion de funciones internas utiles
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


#=======================================================================
# [B.1] Funcion de procesar CVs
#=======================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching

def procesar_CVs_ia(
  ruta_zip,
  aspectos,
  cargo_aplica,
  api_key_openAI
  ):
  
  # definir listas entregables 
  lista_nombres = []
  lista_imgs = []
  lista_txts = []

  # arbir archivo
  zip_ref = zipfile.ZipFile(ruta_zip, 'r')

  # iterar en cada archivo interno del .zip
  for archivo in zip_ref.namelist():
    
    print(f'Extrayendo Archivo: {archivo}')
    
    pdf_data = zip_ref.read(archivo)
    
    pdf_documento = fitz.open(stream=pdf_data, filetype='pdf')
    
    # leer cada pagina y acumular texto
    texto_paginas = ''
    for num_pagina in range(len(pdf_documento)):
      pagina = pdf_documento[num_pagina]
      texto = pagina.get_text()  # Extrae el texto de la página
      texto_paginas += texto+'\n'
      
      
    # convertir a imagen 
    pagina = pdf_documento[0]  # Primera página
    pixmap = pagina.get_pixmap()  # Convierte la página a pixmap (imagen)
    imagen = Image.open(io.BytesIO(pixmap.tobytes('png')))  # Convierte pixmap a imagen PIL
    
    # guardar entregables 
    lista_imgs.append(imagen)
    lista_txts.append(texto_paginas)
    lista_nombres.append(archivo)
    
    # Cierra el archivo PDF
    pdf_documento.close()
    
  # crear diccionario de contenido de cada CV
  dic_contenidos = dict(zip(lista_nombres,lista_txts))


  # crear cliente de openai 
  cliente_OpenAI = OpenAI(api_key=api_key_openAI)

  # crear clase de formato de salida segun aspectos ingresados
  class Resumidor_CV(BaseModel):
    lista_aspectos: list[str]
    lista_puntajes: list[int]
    resumen_general: str
    
  # crear lista de aspectos
  aspectos2 = [x.replace('\n','').strip() for x in aspectos.split(',')]


  # definir prompt del sistema 
  prompt_s = f'''
  Eres un evaluador y resumidor de curriculums de postulantes. A partir de un texto de 
  descripcion del postulante debes rescatar ciertos aspectos y retornarlos en formato 
  especificado. Debes retornar una lista con {len(aspectos2)} elementos de texto,
  referente a los siguientes aspectos: {aspectos2}. Asi mismo, debes retornar otra lista 
  de {len(aspectos2)} valores, donde cada valor es una puntuacion entre 0 y 100 de 
  cada aspecto segun su afinidad para el cargo {cargo_aplica}.
  Finalmente, retornar un resumen general del postulante de no mas de 60 palabras.
  '''

  # iterar sobre cada postulantes para aplicar analisis con IA
  df_consolidado_aspectos = pd.DataFrame([])
  df_consolidado_puntajes = pd.DataFrame([])
  for postulante in list(dic_contenidos.keys()):
    
    print('resumiendo con ia caso: '+postulante)

    # definir prompt del usuario  
    prompt_u = f'''
    la descripcion del postulante es la siguiente: {dic_contenidos[postulante]}
    '''
    # print(prompt_u)

    respuesta_ia = cliente_OpenAI.beta.chat.completions.parse(
      model='gpt-4o-2024-08-06',
      messages=[
        {'role': 'system', 'content': prompt_s},
        {'role': 'user', 'content': prompt_u},
        ],
      response_format=Resumidor_CV
      )
      
    respuesta_ia2 = respuesta_ia.choices[0].message.parsed
    # print(respuesta_ia2)


    # volcar resultado a un df: aspectos
    df_aspectos = pd.DataFrame(
      [[postulante]+respuesta_ia2.lista_aspectos+[respuesta_ia2.resumen_general]],
      columns=['Postulante']+aspectos2+['Resumen General']
    )

    # volcar resultado a un df: puntaje
    df_puntajes = pd.DataFrame(
      [[postulante]+respuesta_ia2.lista_puntajes],
      columns=['Postulante']+aspectos2
    )

      
    df_consolidado_aspectos = pd.concat([df_consolidado_aspectos,df_aspectos])
    df_consolidado_puntajes = pd.concat([df_consolidado_puntajes,df_puntajes])
 
  df_consolidado_aspectos = df_consolidado_aspectos.reset_index(drop=True)
  df_consolidado_puntajes = df_consolidado_puntajes.reset_index(drop=True)
  
    
  return df_consolidado_aspectos,df_consolidado_puntajes,lista_imgs
    
  

#=======================================================================
# [B.2] Funcion para crear grafico de radar
#=======================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def grafico_radar(
  df,
  variable_categoria,
  variables_ejes
  ):

  # Mostrar graficamente
  fig = go.Figure()

  categorias = df[variable_categoria]
  variables = df[variables_ejes]

  for i, row in df.iterrows():
    # Cerrar el gráfico añadiendo el primer valor al final de la lista
    valores_r = row.drop(variable_categoria).values.tolist()
    valores_r.append(valores_r[0])  # Cerrar el gráfico añadiendo el primer valor al final

    # Crear el gráfico de radar para cada postulante
    fig.add_trace(go.Scatterpolar(
      r=valores_r,  # Los valores para el radar, con el primer valor repetido al final
      theta=variables.columns.tolist() + [variables.columns[0]],  # Las categorías, cerradas
      fill=None,  # No rellenar el área
      name=row[variable_categoria]  # El nombre de la fila como leyenda
    ))

  # Actualizar el diseño del gráfico para hacerlo más comprensible
  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        range=[0, 100],
        showticklabels=False  # Quitar los labels de los ticks radiales
      ),
      angularaxis=dict(
        showticklabels=True  # Mostrar los ticks de las categorías (angular axis)
      )
    ),
    showlegend=True  # Mostrar leyenda
  )
  
  return fig







#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [C] Generacion de la App
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

st.set_page_config(layout='wide')

# titulo inicial 
st.markdown('## :page_with_curl: Analizador de CVs con IA :page_with_curl:')

# autoria 
st.sidebar.markdown('**Autor :point_right: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')

# ingresar OpenAI api key
usuario_api_key = st.sidebar.text_input(
  label='Tu OpenAI API key :key:',
  placeholder='Pega aca tu openAI API key',
  type='password'
  )


# subir archivo zip 
archivos_cvs = st.sidebar.file_uploader(
  'Sube un archivo .zip con los CVs en pdf comprimidos', 
  type='zip'
  )


# ingresar listado de links separados por coma
aspectos = st.sidebar.text_area(
  'Ingresa aca los aspectos del CV a revisar separados por ","',
  )


col1, col2 = st.sidebar.columns([2, 1])  # Ajustar proporciones si es necesario

# ingresar cargo al que postula 
cargo_aspira = col1.text_input(
  'Cargo al que postula',
  value = 'Analista de datos'
  )


# colocar boton de procesar 
col2.markdown('### ')
boton_procesar = col2.button('Analizar CVs')

#_____________________________________________________________________________
# comenzar a desplegar app una vez ingresado el archivo

if archivos_cvs is not None and boton_procesar and len(usuario_api_key)>0 and len(cargo_aspira)>0:
  
  df_resumen_CVs, df_puntaje_CVs, lista_imagenes = procesar_CVs_ia(
    ruta_zip = archivos_cvs,
    aspectos = aspectos,
    cargo_aplica = cargo_aspira,
    api_key_openAI = usuario_api_key
    )
  
  # Crear tres tabs
  tab1, tab2, tab3 = st.tabs([
    ':newspaper: Visualizar Curriculums', 
    ':date: Cuadro resumen', 
    ':bar_chart: Puntaje por aspecto'
    ])
  
  
  #...........................................................................
  # Titulo de CVs
  
  with tab1:  
  
    # st.markdown('### 1. Visualizar Curriculums')
    
    # Mostrar CVs
    for i, img in enumerate(lista_imagenes):
      # Expander para cada imagen
      with st.expander(list(df_resumen_CVs['Postulante'])[i]):
        
        col1, col2, col3 = st.columns([1,3,1])  # Crear columnas para centrar el contenido
        with col2:  # Usar la columna del medio
        
          # Mostrar la imagen dentro del expander
          fig, ax = plt.subplots(figsize=(6, 4))
          ax.imshow(img)  # Cambia el cmap si lo prefieres
          ax.axis('off')  # Desactivar ejes

          # Mostrar la figura con plt
          st.pyplot(fig)


  #...........................................................................
  # Procesar CVs y mostrar resumen
  
  with tab2:
     
    st.dataframe(df_resumen_CVs,hide_index=True)


  #...........................................................................
  # Mostrar puntajes y grafico asociado
  
  with tab3:
  
    # st.markdown('### 3. Puntaje en cada aspecto segun cargo al que postula')    
    fig_radar = grafico_radar(
      df = df_puntaje_CVs,
      variable_categoria = 'Postulante',  
      variables_ejes = [x for x in df_puntaje_CVs.columns if x!='Postulante']
      )

    st.dataframe(df_puntaje_CVs,hide_index=True)
    st.plotly_chart(fig_radar)



# !streamlit run App_Analizador_CVs3.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Actividades Seba/Programacion Python/30_Streamlit App chat URL (01-03-24)/"

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit
