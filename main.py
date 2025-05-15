import post_process.vtk_processing as vtk
import post_process.hdf5_processing as hdf
import post_process.ploting as ploting
import post_process.data_processing as data
import os
import json

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # 1) read setting file
    # 2) check if numpy data
    #     2.1) save numpy data from vtk if not
    #     2.2) save numpy data from hdf5 if not
    # 3) generate graphs

    # ns = 3
    # FieldOutputTag = B + E
    # MomentsOutputTag = rho
    # ParticlesOutputTag = position + velocity

    ns = 3
    field_output_E = True
    field_output_Ji = False
    par_output_velocity = True

    multi_var = ["velocity_1_x", "velocity_2_x", "velocity_3_x"]
    # multi_var = []
    single_var = ["Efield_x", "rhoe0", "energy_ele"]
    # 'rho_e'
    all = multi_var+single_var

    # all = ["Efield_x"]
    # single_var = all
    # var = vars[6]
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


