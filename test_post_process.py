"""
test_post_process.py - Testovací sada pro ověření refaktoringu postprocessingu iPIC3D.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import unittest
import numpy as np

# Přidání cesty pro importy
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import unit_convert
from post_process.config_manager import to_bool, load_config, load_plot_config, get_cache_dir, set_active_config
import post_process.read_files as read_files
import post_process.ploting as ploting
import post_process.data_processing as data_processing
from post_process.plasma_units import PlasmaUnits, get_default_units
from post_process import cli
import main


class TestConfigManager(unittest.TestCase):
    """Testy správce konfigurací a pomocných funkcí."""

    def test_to_bool(self):
        self.assertTrue(to_bool(True))
        self.assertTrue(to_bool("true"))
        self.assertTrue(to_bool("True"))
        self.assertTrue(to_bool("1"))
        self.assertTrue(to_bool(1))
        self.assertTrue(to_bool("yes"))
        self.assertTrue(to_bool("on"))

        self.assertFalse(to_bool(False))
        self.assertFalse(to_bool("false"))
        self.assertFalse(to_bool("False"))
        self.assertFalse(to_bool("0"))
        self.assertFalse(to_bool(0))
        self.assertFalse(to_bool("no"))
        self.assertFalse(to_bool(None))

    def test_load_configs(self):
        cfg = load_config()
        self.assertIsInstance(cfg, dict)
        self.assertIn("result_folder", cfg)

        plot_cfg = load_plot_config()
        self.assertIsInstance(plot_cfg, dict)
        self.assertIn("single", plot_cfg)
        self.assertIn("multi", plot_cfg)

    def test_cache_dir(self):
        cache_dir = get_cache_dir()
        self.assertTrue(os.path.exists(cache_dir))


class TestUnitConvert(unittest.TestCase):
    """Testy škálování polí a seznamů."""

    def test_rescale_list(self):
        # Python list
        l = [1.0, 2.0, 3.0]
        res = unit_convert.rescale_list(l, 2.0)
        self.assertEqual(res, [2.0, 4.0, 6.0])

        # NumPy array
        arr = np.array([1.0, 2.0, 3.0])
        res_arr = unit_convert.rescale_list(arr, 3.0)
        np.testing.assert_array_equal(res_arr, np.array([3.0, 6.0, 9.0]))


class TestPlasmaUnits(unittest.TestCase):
    """Testy nového modulu pro fyzikální jednotky a normalizaci."""

    def test_default_units(self):
        pu = get_default_units()
        self.assertGreater(pu.ion.get_plasma_frequency(), 0)
        self.assertGreater(pu.electron.get_plasma_frequency(), 0)
        self.assertGreater(pu.debye_len, 0)
        self.assertGreater(pu.c1.vel_const, 0)
        self.assertGreater(pu.c1.energy_const, 0)

    def test_inp_calibration(self):
        test_inp = os.path.join(CURRENT_DIR, "pre_process", "test_roundtrip.inp")
        if os.path.exists(test_inp):
            pu = PlasmaUnits.from_inp_file(test_inp)
            self.assertAlmostEqual(pu.B_SI * 1e9, 10.0, places=2)
            self.assertAlmostEqual(pu.ion.conc * 1e-6, 10.0, places=2)


class TestReadFiles(unittest.TestCase):
    """Testy nízkoúrovňových pomocných funkcí v read_files.py."""

    def test_parse_text_and_number(self):
        txt, num = read_files.parse_text_and_number("cycle_100")
        self.assertEqual(txt, "cycle")
        self.assertEqual(num, 100)

        txt2, num2 = read_files.parse_text_and_number("proc15")
        self.assertEqual(txt2, "proc")
        self.assertEqual(num2, 15)

    def test_fft_2d(self):
        # Syntetická data: 2D Gaussovský balík
        nx, ny = 32, 16
        x = np.linspace(-5, 5, nx)
        y = np.linspace(0, 10, ny)
        X, Y = np.meshgrid(x, y)
        E = np.exp(-(X**2 + Y**2))

        fft_res = read_files.fft_2d(E, x, y)
        self.assertIsNotNone(fft_res)
        E_fft, kx, ky = fft_res
        self.assertEqual(E_fft.shape, (ny, nx))
        self.assertEqual(len(kx), nx)
        self.assertEqual(len(ky), ny)


class TestPloting(unittest.TestCase):
    """Testy vizualizačního modulu."""

    def test_plot_description(self):
        descr = ploting.PlotDescription(title="Test", label_x="X", label_y="Y")
        descr.set_ylim(-1.0, 1.0)
        descr.set_xlim(0.0, 10.0)
        descr.multidata_labels(["S1", "S2"])

        self.assertEqual(descr.title, "Test")
        self.assertEqual(descr.y_min, -1.0)
        self.assertEqual(descr.y_max, 1.0)
        self.assertEqual(descr.data_labels, ["S1", "S2"])

    def test_plot_data_save(self):
        tmp_plot = os.path.join(CURRENT_DIR, "test_tmp_plot.png")
        x = np.linspace(0, 1, 50)
        y = np.sin(2 * np.pi * x)
        descr = ploting.PlotDescription(title="Sine Test", label_x="Time [s]", label_y="Amp")

        ploting.plot_data(x, y, descr, save_to_file=True, file_name=tmp_plot)
        self.assertTrue(os.path.exists(tmp_plot))

        if os.path.exists(tmp_plot):
            os.remove(tmp_plot)


class TestDataProcessing(unittest.TestCase):
    """Testy datového zpracování a převodů jednotek."""

    def test_quantity_definitions(self):
        self.assertIn("Efield", data_processing.QUANTITY_DEFINITIONS)
        self.assertIn("Bfield", data_processing.QUANTITY_DEFINITIONS)
        self.assertIn("velocity_1", data_processing.QUANTITY_DEFINITIONS)
        self.assertIn("energy_total", data_processing.QUANTITY_DEFINITIONS)

        # Kontrola synchronizace zpětně kompatibilních slovníků
        self.assertEqual(len(data_processing.g_axis_name), len(data_processing.QUANTITY_DEFINITIONS))
        self.assertEqual(len(data_processing.g_axis_si_units), len(data_processing.QUANTITY_DEFINITIONS))

    def test_extract_unit_from_label(self):
        unit_str = data_processing.extract_unit_from_label("Electric field [V/m]")
        self.assertEqual(unit_str, "V/m")

        unit_str2 = data_processing.extract_unit_from_label("Time [s]")
        self.assertEqual(unit_str2, "s")

    def test_fft_data_transform(self):
        t = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * 10 * t)
        freqs, mag = data_processing.fft_data_transform(t, y)

        self.assertGreater(len(freqs), 0)
        self.assertEqual(len(freqs), len(mag))
        peak_freq = freqs[np.argmax(mag)]
        self.assertAlmostEqual(peak_freq, 10.0, delta=1.5)


class TestCLI(unittest.TestCase):
    """Testy příkazové řádky (CLI)."""

    def test_build_parser(self):
        parser = cli.build_parser()
        args = parser.parse_args(["-d", "/test/data", "-o", "/test/out", "-v", "Efield_x", "--show"])
        self.assertEqual(args.data_dir, "/test/data")
        self.assertEqual(args.output_dir, "/test/out")
        self.assertEqual(args.single_vars, ["Efield_x"])
        self.assertTrue(args.show_plots)

    def test_parse_plot_types(self):
        d = cli.parse_plot_types("val,fft2d,hist")
        self.assertEqual(d["Value"], "True")
        self.assertEqual(d["FFT2D"], "True")
        self.assertEqual(d["Histogram"], "True")
        self.assertEqual(d["FFT"], "False")

    def test_cli_list_vars(self):
        ret = cli.main(["--list-vars"])
        self.assertEqual(ret, 0)


class TestStandaloneIsolation(unittest.TestCase):
    """
    Klíčový test přenositelnosti na server:
    Zkopíruje POUZE složku post_process do izolovaného dočasného adresáře
    a ověří, že z něj lze bez chyb spustit CLI (žádné skryté vnější závislosti).
    """

    def test_isolated_execution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_pkg = os.path.join(CURRENT_DIR, "post_process")
            dst_pkg = os.path.join(tmp_dir, "post_process")
            shutil.copytree(src_pkg, dst_pkg)

            # Spuštění python -m post_process --list-vars v izolovaném adresáři
            res = subprocess.run(
                [sys.executable, "-m", "post_process", "--list-vars"],
                cwd=tmp_dir,
                capture_output=True,
                text=True
            )
            self.assertEqual(res.returncode, 0, f"Selhalo v izolovaném prostředí: {res.stderr}")
            self.assertIn("Efield_x", res.stdout)
            self.assertIn("energy_ele", res.stdout)


if __name__ == "__main__":
    unittest.main()
