from imwatermark import WatermarkEncoder, WatermarkDecoder
from gooey import Gooey, GooeyParser
from pathlib import Path
import cv2


def encode(file):
    encoder = WatermarkEncoder()
    wm = "StableDiffusionV1"
    encoder.set_watermark("bytes", wm.encode("utf-8"))
    return encoder.encode(file, "dwtDct")


def decode(file, wtr):
    decoder = WatermarkDecoder("bytes", 8 * len(wtr))
    wm = decoder.decode(file, "dwtDct")
    return wm.decode("utf-8")


@Gooey(
    program_name="ShoddyWater", advanced=True, tabbed_groups=True, clear_before_run=True
)
def main():
    parser = GooeyParser(
        description='Program to add "StableDiffusionV1" watermark to your images to prevent scraping.'
    )
    required_args = parser.add_argument_group("Required Arguments")
    opt_args = parser.add_argument_group("Optional Arguments")
    required_args.add_argument(
        "-op",
        "--operation",
        help="What operation",
        choices=["Encode", "Decode"],
        required=True,
    )
    required_args.add_argument(
        "-f",
        "--file",
        help="The file to do the operation on.",
        widget="FileChooser",
        required=True,
    )
    opt_args.add_argument(
        "-v",
        "--verbose",
        help="Verbose/debug output?",
        action="store_true",
        required=False,
    )
    opt_args.add_argument(
        "-o",
        "--out",
        help='The file to output to. (Default is to the same directory with "_swtr" suffix, ignored for decode)',
        widget="FileSaver",
        required=False,
    )
    args = parser.parse_args()
    v = args.verbose
    op = args.operation

    if op == "Encode":
        if v:
            print("Embedding into image...")
        embedded = encode(cv2.imread(args.file))
        if v:
            print("Embedded.")

        if v:
            print("Writing image to output path...")
        path = (
            args.out
            or Path.joinpath(
                Path(args.file).parent,
                f"{Path(args.file).stem}_swtr{Path(args.file).suffix}",
            ).__str__()
        )
        if args.out:
            cv2.imwrite(path, embedded)
        else:
            cv2.imwrite(path, embedded)
        print(f"Image written to: {path}")

    if op == "Decode":
        if v:
            print("Decoding image...")
        decoded = decode(cv2.imread(args.file), "StableDiffusionV1")
        if v:
            print("Decoded.")

        if v:
            print(f'Checking if watermark is in "{args.file}"...')
        if v:
            print(f"Watermark is: {decoded}")
        if decoded == "StableDiffusionV1":
            print("Watermark is in image!")
        else:
            print("Watermark is NOT in image!")


if __name__ == "__main__":
    main()
