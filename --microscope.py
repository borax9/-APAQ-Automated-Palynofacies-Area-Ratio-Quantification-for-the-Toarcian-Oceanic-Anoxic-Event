import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="PalyAI Analytical Engine (Phase 2)")
    # Kullanıcının mikroskop seçmesini sağlayan parametre (Varsayılan: olympus)
    parser.add_argument('--microscope', type=str, default='olympus', choices=['olympus', 'nikon'],
                        help="Select the microscope platform used for image acquisition ('olympus' or 'nikon')")
    return parser.parse_args()

def get_calibration_factors(microscope_type):
    # Her mikroskobun kendi donanımına özel piksel-mikron katsayıları
    calibration_database = {
        'olympus': {
            '10x': 1.2403,
            '20x': 0.6202,
            '40x': 0.3101
        },
        'nikon': {
            '10x': 1.1520,  # Nikon + Jenoptik için ölçtüğünüz/hesapladığınız gerçek değerler
            '20x': 0.5760,  # (Buradaki değerleri kendi ölçümünüze göre güncelleyebilirsiniz)
            '40x': 0.2880
        }
    }
    return calibration_database.get(microscope_type, calibration_database['olympus'])

# Ana akışta kullanımı:
args = parse_arguments()
cal_factors = get_calibration_factors(args.microscope)

# Alan hesaplama aşamasında (Örn: 10x için):
# absolute_area = pixel_count * (cal_factors['10x'] ** 2)