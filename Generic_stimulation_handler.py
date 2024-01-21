import pandas as pd
import numpy as np
from scipy.stats import mode
from typing import List, Tuple, Union, Optional, Dict, Any
import matplotlib.pyplot as plt

from Generic_foldering_operations import *
from Generic_string_operations import *
from Generic_numeric_operations import *

class stimulation_data:
    def __init__(self, path: str, Stim_var: str = 'Orientamenti', Time_var: str = 'N_frames', phys_recording_type: str = 'F', correct_stim_duration: Union[str, int] = 'mode'):

        """
        Classe per gestire i dati di stimolazione da file Excel.

        Parameters:
        - path (str): Percorso della directory contenente i file Excel.
        - Stim_var (str): Nome della colonna che contiene il tipo di stimolo.
        - Time_var (str): Nome della colonna che contiene il tempo dell'onset di stimolo.
        - phys_recording_type (str, optional): Type of physiological recording. Default is 'F' (i.e. fluorescence 2p).
        - correct_stim_duration (Union[str, int], optional):
          The desired duration of the stimulus. 'mode' to use mode of durations, or an integer value. Default is 'mode'.
        """
        
        self.path = path
        self.Stim_var = Stim_var
        self.Time_var = Time_var
        self.correct_stim_duration = correct_stim_duration #da riflettere se mantenere nell'inizializzazione o in metodi successivi
        self.phys_recording_type = phys_recording_type

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
    
    def get_len_phys_recording(self, stimulation_df: pd.DataFrame) -> List[Union[int, float]]: #controlla per dati diversi da Fluorescenza 2p
        """
        Ottieni la lunghezza della registrazione fisiologica dai dati di stimolazione.

        Parameters:
        - stimulation_df (pd.DataFrame): DataFrame contenente i dati di stimolazione.

        Returns:
        - len_phys_recording (List[Union[int, float]]): Lunghezza della registrazione fisica.
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
        session_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in excel_files] #os.path.splitext(os.path.basename(file_path))[0] prendo solo il root, perdendo la extension
        Stim_dfs = []
        StimVecs = []
        for n,ex_f in enumerate(excel_files): #multipli file excel sono considerati in caso di trattamento nella stessa sessione sperimentale (pre-post treatment). L'ordinamento di questi files va sviluppato
         Stim_dfs.append(pd.read_excel(ex_f))
         StimVecs.append(self.get_StimVec(Stim_dfs[n]))
        len_phys_recordings = self.get_len_phys_recording(Stim_dfs[n])
       
        self.Stim_dfs = Stim_dfs
        self.StimVecs = StimVecs
        self.len_phys_recordings = len_phys_recordings
        self.session_names = session_names
        return Stim_dfs, StimVecs, len_phys_recordings, session_names
    
    def create_logical_dict(self, idx: int, change_existing_dict_files: bool=True)-> Dict[str, Any]: #mettere anche *args e **kwargs?
      """
      Create a logical dictionary containing the timings on each stimulus type.

      Parameters:
      - idx (int): Index specifying which StimVec and Stim_df to use.
      - change_existing_dict_files (bool, optional): Flag to change existing dictionary files. Default is True.

      Returns:
      Dict[str, Any]: The logical dictionary containing information about stimuli timings.
      """
      session_name = self.path; Stim_var = self.Stim_var
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
    

    def get_stim_phys_recording(self, stim_name: str, phys_recording: np.ndarray, idx_logical_dict: int = 0, get_pre_stim: bool = False, correct_stim_duration: int = 0, latency=0) -> np.ndarray:
      """
      Retrieves the physiological recordings corresponding to each occurrence of a stimulus.

      Parameters:
      - stim_name (str): The name of the stimulus.
      - phys_recording (np.ndarray): Array of physiological recordings.
      - idx_logical_dict (int, optional):
        useful in case of multiple logical dicts present. Set to 0 for single sessions

      Returns:
      np.ndarray: Array containing the stimulus' physiological recordings.
      """
      stimTrue_begin_end = self.logical_dict[idx_logical_dict][stim_name]; stim_durations = stimTrue_begin_end[:, 1] - stimTrue_begin_end[:, 0]
      if correct_stim_duration == 0:
        correct_stim_duration = self.correct_stim_duration
        if correct_stim_duration == 'mode':
          correct_stim_duration = int(mode(stim_durations)[0]) #si assume che la moda delle durate sia la durata normale dello stimolo
        else:
          correct_stim_duration = int(correct_stim_duration) 
        
      stim_phys_recordings = np.full((stimTrue_begin_end.shape[0], phys_recording.shape[0],correct_stim_duration), np.nan)
      for i, stim_event_beg_end in enumerate(stimTrue_begin_end):
        Sev_begin = stim_event_beg_end[0]
        is_duration_correct = np.abs(stim_durations[i]-int(mode(stim_durations)[0]))< int(mode(stim_durations)[0])/10 #criterio arbitrario
        is_phys_registered = phys_recording.shape[1] >= stimTrue_begin_end[i, 1] #l'evento è stato registrato per intero fisiologicamente
        if is_duration_correct and is_phys_registered:
           if get_pre_stim:
              Sev_begin = Sev_begin-latency
              stim_phys_recordings[i,:,:] = phys_recording[:,Sev_begin-correct_stim_duration:Sev_begin]
           else:
            Sev_begin = Sev_begin+latency
            stim_phys_recordings[i,:,:] = phys_recording[:,Sev_begin:Sev_begin+correct_stim_duration]
      return stim_phys_recordings
    
    def get_stats(self, phys_recording: np.ndarray, functions_to_apply, n_it: int =0,change_existing_dict_files: bool=True):
       results = []
       for fun in functions_to_apply:
          results.append(fun(self,phys_recording,n_it,change_existing_dict_files))
       return results


def get_stims_mean_sem(stimulation_data_obj, phys_recording: np.ndarray, n_it: int =0, change_existing_dict_files: bool=True) -> Dict[str, Any]:
    """
    Calculate mean and SEM for each stimulus type and save the results.

    Parameters:
    - stimulation_data_obj: instance of the stimulation_data class
    - phys_recording (np.ndarray): Array of physiological recordings.
    - n_it (int, optional): Index specifying which logical dictionary to use. Default is 0.
    - change_existing_dict_files (bool, optional): Flag to change existing dictionary files. Default is True.

    Returns:
    Dict[str, Any]: Dictionary containing mean and SEM values for each stimulus type.
    """
    session_name = os.path.basename(stimulation_data_obj.path); logical_dict = stimulation_data_obj.logical_dict[n_it]; phys_recording_type = stimulation_data_obj.phys_recording_type
    #phys_recording_type can be set to F, Fneu, F_neuSubtract, DF_F, DF_F_zscored
    Mean_SEM_dict_filename = session_name+phys_recording_type+'_Mean_SEM_dict.npz'
    if not(os.path.isfile(Mean_SEM_dict_filename)) or change_existing_dict_files==True:
        Mean_SEM_dict = {}
        for key in logical_dict.keys():
          stim_phys_recordings = stimulation_data_obj.get_stim_phys_recording(key, phys_recording, idx_logical_dict=n_it)
          mean_betw_cells = np.mean(stim_phys_recordings, axis = 1)
          Mean = np.mean(mean_betw_cells, axis=0)
          if stim_phys_recordings.shape[0]==1:
            SEM = SEMf(stim_phys_recordings[0,:,:]) #sem between cells for stimuli that are presented only once
          else:
            SEM = SEMf(mean_betw_cells)
          Mean_SEM_dict[key] = np.column_stack((Mean, SEM))
        np.savez(Mean_SEM_dict_filename, **Mean_SEM_dict)
    else:
        Mean_SEM_dict = np.load(Mean_SEM_dict_filename)
    return Mean_SEM_dict
    
def cut_recording(StimVec: np.ndarray,Stim_df: pd.DataFrame, physRecordingMatrices: List[np.ndarray], df_Time_var: str, 
                  do_custom_cutting: Optional[bool] = False) -> Tuple[np.ndarray, pd.DataFrame, List[np.ndarray]]:
  """
  Cut the recording data based on custom cutting.

  Parameters:
  - StimVec (np.ndarray): The vector representing stimuli in each timebin.
  - Stim_df (pd.DataFrame): DataFrame containing stimuli and time related information.
  - physRecordingMatrices (List[np.ndarray]): List of matrices containing physiological recording data.
  - df_Time_var (str): Column name representing time variable in Stim_df.
  - do_custom_cutting (bool, optional): Flag to enable custom cutting. Default is False.

  Returns:
  Tuple[np.ndarray, pd.DataFrame, List[np.ndarray]]: Tuple containing cut StimVec, cut Stim_df, and cut physRecordingMatrices.
  """
    
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

               
