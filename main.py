import post_process.vtk_processing as vtk
import post_process.hdf5_processing as hdf
import post_process.ploting as ploting
import post_process.data_processing as data
import os

def check_saved_data(var):
    script_data_folder = "script_data"
    if any(os.listdir(script_data_folder)):  # Check if folder contains any files or subfolders
        print("The folder is not empty.")
        files = os.listdir(script_data_folder)
        print(files)

        found = any(var in f for f in files)  # Check if any file contains vars[0]

        print("Founded" if found else "Not founded")
        # vtk.result_analysis(var)
        if not found:
            try:
                try:
                    vtk.result_analysis(var)
                except:
                    hdf.result_analysis(var)
                print("Data loaded")
            except:
                print(f"Data with name {var} not available")


    else:
        print("The folder is empty.")
        try:
            vtk.result_analysis(var)
        except:
            hdf.result_analysis(var)
        print("Data loaded")

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

    vars = ["velocity_1_x", "velocity_2_x", "velocity_3_x", "Efield_x", 'rho_e']

    var = vars[3]
    print(f">> chosen data: {var}")
    check_saved_data(var)

    data.single_result_analysis(var)
    # data.multi_result_analysis([vars[0], vars[1], vars[2]])

    # result_type = "hdf11"

    # vars = ["E", "B", "rhoe0", "rhoe2", "rhoi1", "rhoi3"]
    # vars = ["rhoe2", "rhoi3"]
    # vars = ["E"]

    # print("ANALYSING START")
    # match result_type:
    #     case "vtk":
    #         for v in vars:
    #             vtk.result_analysis(v)
    #             # try:
    #             #     result_analysis(v)
    #             # except:
    #             #     print(f"file not found for {v} variable")
    #     case "hdf":
    #         hdf.result_analysis("v")
    #     case "both":
    #         hdf.result_analysis("v")
    #         vtk.result_analysis("E")
    #     case "comp":
    #         hdf.compare_result()
    #     case _:
    #         print(f"Non valid result type: {result_type}")
    #
    ploting.plot_all_graphs()

    print("ANALYSING DONE")


