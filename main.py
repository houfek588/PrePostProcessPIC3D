import post_process.vtk_processing as vtk
import post_process.hdf5_processing as hdf
import post_process.ploting as ploting

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    result_type = "vtk"

    # vars = ["E", "B", "rhoe0", "rhoe2", "rhoi1", "rhoi3"]
    # vars = ["rhoe2", "rhoi3"]
    vars = ["E"]

    print("ANALYSING START")
    match result_type:
        case "vtk":
            for v in vars:
                vtk.result_analysis(v)
                # try:
                #     result_analysis(v)
                # except:
                #     print(f"file not found for {v} variable")
        case "hdf":
            hdf.result_analysis()
        case "both":
            hdf.result_analysis()
            vtk.result_analysis("E")
        case "comp":
            hdf.compare_result()
        case _:
            print("Non valid result type")

    # ploting.plot_all_graphs()

    print("ANALYSING DONE")
