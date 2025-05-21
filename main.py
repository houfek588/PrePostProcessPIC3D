import post_process.vtk_processing as vtk
import post_process.hdf5_processing as hdf
import post_process.ploting as ploting
import post_process.data_processing as data
import os
import json

g_available = ["Efield_x", "Efield_y", "Efield_z", "rhoe0", "rhoi1", "rhoe2", "velocity_1", "velocity_2",
             "velocity_3", "energy_kin", "energy_ele"]

def check_saved_data(var):
    script_data_folder = "script_data"
    if any(os.listdir(script_data_folder)):  # Check if folder contains any files or subfolders
        print("The folder is not empty.")
        files = os.listdir(script_data_folder)
        # print(files)

        found = any(var in f for f in files)  # Check if any file contains vars[0]

        print("Founded" if found else "Not founded")
        # vtk.result_analysis(var)
        if not found:
            try:
                try:
                    vtk.result_analysis(var)
                    print(" vtk data loaded")
                except:
                    hdf.result_analysis(var)
                    print(" hdf data loaded")
                # print("Data loaded")
            except:
                try:
                    vtk.conserve_analysis(var)
                    print(" vtk conserve data loaded")
                except:
                    raise Exception(f"Data with name {var} not available")


    else:
        print("The folder is empty.")
        try:
            vtk.result_analysis(var)
        except:
            hdf.result_analysis(var)
        print("Data loaded")

def clear_script_data():
    # Load configuration from JSON file
    with open("config.json", "r") as file:
        parameters = json.load(file)

    # Extract relevant parameters
    # folder_path = parameters["result_folder"]
    folder_path = "script_data/"

    # Loop over files and remove them
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Remove only files (not folders)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"deleted: {file_path}")

def data_process(single_var, multi_var):

    # multi_var = ["velocity_1_x", "velocity_2_x", "velocity_3_x"]
    # multi_var = ["velocity_1_x", "velocity_3_x"]
    # multi_var = []
    # single_var = ["Efield_x", "rhoe0", "energy_ele"]
    # single_var = ["Efield_x", "energy_ele"]
    # single_var = []
    # 'rho_e'
    all = multi_var + single_var

    print(f">> chosen data: {all}")

    for one in all:
        check_saved_data(one)

    # numpy data analysis to graphs
    for s in single_var:
        data.single_result_analysis(s)
    data.multi_result_analysis(multi_var)

    # clear saved numpy data
    # clear_script_data()

    # /lib64/mpich/bin/mpiexec -n 32 ./iPIC3D inputfiles/file_name.inp

    ploting.plot_all_graphs()

    print("ANALYSING DONE")

def save_json(json_data, file_name: str = "confin.json"):
    with open(file_name, "w") as f:
        f.write(json_data)
    print(f"Configuration succesfully writen into {file_name}")

def save_config(result_folder, output_folder, save_graphs, FieldOutputCycle, ParticleOutputCycle, NumberHdfFiles):
    data = {
        "result_folder": result_folder,
        "output_folder": output_folder,
        "save_graphs": str(save_graphs),
        "FieldOutputCycle": FieldOutputCycle,
        "ParticleOutputCycle": ParticleOutputCycle,
        "NumberHdfFiles": NumberHdfFiles
    }

    json_data = json.dumps(data)
    file_name = "config.json"
    save_json(json_data, file_name)


def save_plot_config(options_single, options_multi):
    print(f"options_single: {options_single}")
    print(f"options_multi: {options_multi}")


    data = {
        "single": options_single,
        "multi": options_multi,
    }

    json_data = json.dumps(data)
    file_name = "plot_config.json"
    save_json(json_data, file_name)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # 1) read setting file
    # 2) check if numpy data
    #     2.1) save numpy data from vtk if not
    #     2.2) save numpy data from hdf5 if not
    # 3) generate graphs


    # multi_var = ["velocity_1_x", "velocity_2_x", "velocity_3_x"]
    # multi_var = ["velocity_1_x", "velocity_3_x"]
    multi_var = []
    # single_var = ["Efield_x", "rhoe0", "energy_ele"]
    # single_var = ["Efield_x", "energy_ele"]
    single_var = ["rhoe2"]
    # single_var = []
    # 'rho_e'

    data_process(single_var, multi_var)

    # data.ReadPlotConfig("plot_config.json")


