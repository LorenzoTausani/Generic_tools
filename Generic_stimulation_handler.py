import pandas as pd
import numpy as np
from typing import List, Tuple, Union
import matplotlib.pyplot as plt

from Generic_foldering_operations import *
from Generic_string_operations import *


class stimulation_data:
    def __init__(self, path: str, Stim_var: str = 'Orientamenti', Time_var: str = 'N_frames'):

        """
        Classe per gestire i dati di stimolazione da file Excel.

        Parameters:
        - path (str): Percorso della directory contenente i file Excel.
        - Stim_var (str): Nome della colonna che contiene il tipo di stimolo.
        - Time_var (str): Nome della colonna che contiene il tempo dell'onset di stimolo.
        """
        
        self.path = path
        self.Stim_var = Stim_var
        self.Time_var = Time_var

    def Stim_var_rename(self, stimulation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fa eventuali operazioni di rinomina degli stimoli (utilizzato, CL). Di solito è custom sull'esperimento

        Parameters:
        - stimulation_df (pd.DataFrame): DataFrame contenente i dati di stimolazione (stims e onset times).

        Returns:
        - stimulation_df (pd.DataFrame): DataFrame con la colonna degli stimoli rinominata.
        """
        for it, row in stimulation_df.iterrows():
          it_stim = row[self.Stim_var]
          stimulation_df[self.Stim_var][it] = str(it_stim)
        return stimulation_df
    
    def get_StimVec(self, stimulation_df: pd.DataFrame) -> np.ndarray:
        """
        Ottieni un vettore che contiene, per ogni bin temporale, il tipo di stimolazione utilizzata

        Parameters:
        - stimulation_df (pd.DataFrame): DataFrame contenente i dati di stimolazione.

        Returns:
        - StimVec (np.ndarray): Vettore di stimoli.
        """
        stimulation_df = self.Stim_var_rename(stimulation_df)
        StimVec = np.empty(stimulation_df[self.Time_var].max(), dtype=object) # Crea un array vuoto per il vettore di stimoli

        # Itera sul DataFrame e assegna il tipo di stimolo a ogni unità di tempo
        top=0
        for it, row in stimulation_df.iterrows():
          if it==0:
            prec_row = row
          else:
            StimVec[top:row[self.Time_var]] = prec_row[self.Stim_var]
            top=row[self.Time_var]
            prec_row = row
        return StimVec
    
    def get_len_phys_recording(self, stimulation_df: pd.DataFrame) -> Union[int, float]:
        """
        Ottieni la lunghezza della registrazione fisiologica dai dati di stimolazione.

        Parameters:
        - stimulation_df (pd.DataFrame): DataFrame contenente i dati di stimolazione.

        Returns:
        - len_phys_recording (Union[int, float]): Lunghezza della registrazione fisica.
        """
        return [stimulation_df[self.Time_var][-1]]

    def get_stim_data(self) -> Tuple[List[pd.DataFrame], List[np.ndarray], List[Union[int, float]]]:
       
        """
        processing dei dati di stimolazione

        Returns:
        - Stim_dfs (List[pd.DataFrame]): Lista di DataFrame contenenti i dati di stimolazione.
        - StimVecs (List[np.ndarray]): Lista di vettori di stimoli.
        - len_phys_recordings (List[Union[int, float]]): Lista di lunghezze delle registrazioni fisiologiche durante la stimolazione.
        """

        excel_files = find_files_by_extension(directory = self.path, extension='.xlsx', recursive = False)
        Stim_dfs = []
        StimVecs = []
        for n,ex_f in enumerate(excel_files): #multipli file excel sono considerati in caso di trattamento nella stessa sessione sperimentale (pre-post treatment). L'ordinamento di questi files va sviluppato
         Stim_dfs.append(pd.read_excel(ex_f))
         StimVecs.append(self.get_StimVec(Stim_dfs[n]))
        len_phys_recordings = self.get_len_phys_recording(Stim_dfs[n])
       
        self.Stim_dfs = Stim_dfs
        self.StimVecs = StimVecs
        self.len_phys_recordings = len_phys_recordings
        return Stim_dfs, StimVecs, len_phys_recordings
    
    def create_logical_dict(self, idx, change_existing_dict_files=True): #mettere anche *args e **kwargs?
      session_name = self.session_name; Stim_var = self.Stim_var
      StimVec = self.StimVecs[idx]; df = self.Stim_dfs[idx]
      stim_names = df[Stim_var].unique()
      logical_dict_filename = session_name+'_logical_dict.npz'; 
      if not(os.path.isfile(logical_dict_filename)) or change_existing_dict_files==True:
          logical_dict ={}
          for stim in stim_names:
            if stim != 'END':
                stimTrue =StimVec==stim # Definisci il vettore booleano di True e False
                stimTrue_01 = ''.join('1' if x else '0' for x in stimTrue) # Converti il vettore booleano in una stringa di 0 e 1
                # Trova tutte le sequenze di 1 consecutive nella stringa e calcola la loro lunghezza
                stimTrue_begin_end = [(match.start(), match.end()) for match in re.finditer('1+', stimTrue_01)]
                logical_dict[str(stim)] = np.array(stimTrue_begin_end) 

          if hasattr(self, 'add_keys_logicalDict') and callable(getattr(self, 'add_keys_logicalDict')):
            logical_dict = self.add_keys_logicalDict(logical_dict)
          np.savez(logical_dict_filename, **logical_dict)
      else:
          logical_dict = np.load(logical_dict_filename)
      if not(hasattr(self, 'logical_dict')):
        self.logical_dict = [logical_dict]
      else:
         self.logical_dict.append(logical_dict)
      return logical_dict 

def cut_recording(StimVec,Stim_df, physRecordingMatrices, df_Time_var, do_custom_cutting = False):
  cut = len(StimVec)
  if do_custom_cutting:
    plt.plot(np.mean(physRecordingMatrices[0],axis = 0))
    # Show the plot
    plt.show()
    plt.pause(0.1)
    cut = int(input('At which frame you want to cut the recording (include all = ' +str(len(StimVec))+ ')?'))
    StimVec = StimVec[:cut]
    Stim_df = Stim_df[Stim_df[df_Time_var]<cut] #taglia fuori END? da controllare
  for i,phyR in enumerate(physRecordingMatrices):
    physRecordingMatrices[i] = phyR[:,:cut]
  return StimVec, Stim_df, physRecordingMatrices

               
