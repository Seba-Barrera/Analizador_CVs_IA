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

# libreria para manipular archivos
import zipfile
import fitz  # PyMuPDF
import io



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [B] Creacion de funciones internas utiles
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


#=======================================================================
# [B.1] Funcion de procesar archivo zip
#=======================================================================



@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def zip2_Img_txt(
  archivo_zip_CVs,
  max_ancho = 800
  ):
  
  # definir listas entregables 
  lista_nombres = []
  lista_imgs = []
  lista_txts = []
  
  # arbir archivp  
  zip_ref = zipfile.ZipFile(io.BytesIO(archivo_zip_CVs.read()), 'r')
  
  # iterar en cada archivo interno del .zip
  for archivo in zip_ref.namelist():
    
    print(f'Procesando Archivo: {archivo}')
    
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


    # ajustar tamaño 
    ancho, alto = imagen.size
    ratio_img = alto / ancho
    ancho2 = max_ancho
    alto2 = int(ancho2 * ratio_img)
    imagen =  imagen.resize((ancho2, alto2))
    
    # guardar entregables 
    lista_imgs.append(imagen)
    lista_txts.append(texto_paginas)
    lista_nombres.append(archivo.replace('.pdf',''))
    
    # Cierra el archivo PDF
    pdf_documento.close()
  
  # retornar entregables 
  return lista_nombres,lista_imgs,lista_txts
  



#=======================================================================
# [B.2] Funcion de procesar CVs
#=======================================================================


@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def procesar_CV(
  _cliente_OpenAI,
  nombre_cv,
  texto_cv,
  aspectos
  ):
 
  # definir rol del sistema 
  rol_sistema = 'Eres un experto analizador y lector de curriculums'

  # escribir prompt
  prompt = f'''
  Eres un experto lector de curriculumns de candidatos, necesito que a partir del 
  siguiente curriculum: 
  "{texto_cv}"
  extraigas las siguientes caracteristicas: {aspectos}.
  retorna tus resultados en una lista de python de {len(aspectos.split(','))} elementos 
  agregando al final un ultimo elemento que sea una descripcion de no mas de 30 palabras 
  del curriculum.
  retorna unicamente la lista en python sin ningun comentario o explicacion.
  '''

  consulta = cliente_OpenAI.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
      {
        'role': 'system', 
        'content': rol_sistema
        },
      {
        'role': 'user',
        'content': prompt
      }
    ]
  )


  # procesar respuesta 
  respuesta = consulta.choices[0].message.content
  respuesta2 = respuesta.replace('\n','')
  respuesta3 = respuesta2[(respuesta2.find('[')):(respuesta2.rfind(']')+1)]


  lista_respuestas = eval(respuesta3)
  
  print(nombre_cv)
  print(lista_respuestas)
  
  try:
  
    lista_respuestas2 = [
      ', '.join(x) if isinstance(x, list) else 
      ', '.join(f'{c}: {v}' for c, v in x.items()) if isinstance(x, dict)
      else str(x) 
      for x in lista_respuestas 
      ]
  
  except Exception as e:
    
    print(f'Ocurrió un error al evaluar la expresión: {e}')
    lista_respuestas2 = [' '] * len([x.strip() for x in aspectos.replace('\n','').split(',')])
    

  print(lista_respuestas2)

  # agregar nombre del archivo en primer lugar 
  lista_respuestas2.insert(0,nombre_cv.replace('.pdf',''))

  cols_df_caso = [x.strip() for x in aspectos.replace('\n','').split(',')]
  cols_df_caso.append('resumen')
  cols_df_caso.insert(0,'CV')

  df_caso = pd.DataFrame([lista_respuestas2], columns=cols_df_caso)
  
  return df_caso



#=======================================================================
# [B.3] Funcion de procesar CVs multiples
#=======================================================================


@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def procesar_CVs(
  _cliente_OpenAI,
  nombres_cv,
  textos_cv,
  aspectos
  ):
  
  # crear df en blanco que sera usado para entregar resultado final
  df_entregable = pd.DataFrame([])
  
  for i in range(len(nombres_cv)):
    
    nom_cv = nombres_cv[i]
    txt_cv = textos_cv[i]
    
    df1 = procesar_CV(
      _cliente_OpenAI = _cliente_OpenAI,
      nombre_cv = nom_cv,
      texto_cv = txt_cv,
      aspectos = aspectos
      )

    # aplicar resultantes
    df_entregable = pd.concat([df_entregable,df1])

  return df_entregable




#=======================================================================
# [B.4] Funcion de asingar puntajes segun caracteristicas de CV
#=======================================================================



@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def puntear_atributos(
  _cliente_OpenAI,
  df_CVs,
  cargo_aplica
  ):
  
  # crear diccionario en blanco donde se iran almacenando respuestas 
  dict_entegable = {
    'CV': list(df_CVs['CV'])
  }

  # crear lista de atributos 
  atributos = [x.strip() for x in df_CVs.columns]
  atributos = atributos[1:-1] # quitar nombre del CV y descripcion 
  
  
  # iterar por cada atributos 
  for atributo in atributos:
    
    valores = list(df_CVs[atributo])

    # definir rol del sistema 
    rol_sistema = 'Eres un experto analizador y evaluador de atributos de curriculums'

    # escribir prompt
    prompt = f'''
    Eres un experto evaluador de curriculumns de candidatos, necesito que considerando el 
    siguiente cargo: "{cargo_aplica}", puntues entre 10 y 100 la caracteristica: "{atributo}"
    para cada uno de los siguientes elementos de la lista: {valores}
    retorna unicamente una lista en python con los valores numericos sin ningun comentario o explicacion.
    '''

    consulta = cliente_OpenAI.chat.completions.create(
      model='gpt-4o-mini',
      messages=[
        {
          'role': 'system', 
          'content': rol_sistema
          },
        {
          'role': 'user',
          'content': prompt
        }
      ]
    )


    # procesar respuesta 
    respuesta = consulta.choices[0].message.content
    respuesta2 = respuesta.replace('\n','')
    respuesta3 = respuesta2[(respuesta2.find('[')):(respuesta2.rfind(']')+1)]
    lista_respuestas = eval(respuesta3)
    
    
    # acumular resultado en diccionario 
    dict_entegable[atributo] = lista_respuestas
  
  df_puntaje = pd.DataFrame(dict_entegable)
  
  
  # crear objeto grafico 
  categorias = df_puntaje.columns.tolist()[1:]
  fig = go.Figure()
  for i, row in df_puntaje.iterrows():
    fig.add_trace(go.Scatterpolar(
      r=row[1:].values.tolist() + [row[1:].values[0]],  # Asegurarse de cerrar el círculo
      theta=categorias + [categorias[0]],  # Repetir la primera categoría para cerrar el círculo
      name=row['CV']  # Nombre de la serie de la primera columna
    ))
  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        range=[0, 100],
        showticklabels=False  # Quitar los labels de los ticks radiales
        )
    ),
    showlegend=True  # Mostrar leyenda
  )
  
  # generar entregable 
  return df_puntaje,fig




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
boton_procesar = col2.button('Analizar CVs',)

#_____________________________________________________________________________
# comenzar a desplegar app una vez ingresado el archivo

if archivos_cvs is not None and boton_procesar and len(usuario_api_key)>0 and len(cargo_aspira)>0:
  
  lista_n, lista_i, lista_t = zip2_Img_txt(
    archivo_zip_CVs = archivos_cvs  
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
    for i, img in enumerate(lista_i):
      # Expander para cada imagen
      with st.expander(lista_n[i]):
        
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
  
    # st.markdown('### 2. Cuadro resumen')

    cliente_OpenAI = OpenAI(api_key=usuario_api_key)
                            
    df_CVs = procesar_CVs(
      _cliente_OpenAI = cliente_OpenAI,
      nombres_cv = lista_n,
      textos_cv = lista_t,
      aspectos = aspectos
      )
    
    df_CVs2 = df_CVs.set_index(df_CVs.columns[0])
    
    st.dataframe(df_CVs2)


  #...........................................................................
  # Mostrar puntajes y grafico asociado
  
  with tab3:
  
    # st.markdown('### 3. Puntaje en cada aspecto segun cargo al que postula')
    
    df_puntaje,fig_puntaje = puntear_atributos(
      _cliente_OpenAI= cliente_OpenAI,
      df_CVs=df_CVs,
      cargo_aplica = cargo_aspira
      )

    df_puntaje2 = df_puntaje.set_index(df_puntaje.columns[0])

    st.dataframe(df_puntaje2)
    st.plotly_chart(fig_puntaje)








# !streamlit run App_Analizador_CVs2.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Actividades Seba/Programacion Python/43_ Analizador de CVs con IA (13-12-24)/App/"

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit
