"""
Generate SiteGiant Import Guide PDF - How to batch update prices in SiteGiant
"""

from pathlib import Path

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

# Paths
SCREENSHOT_DIR = Path(r"C:\Users\User\AppData\Local\Temp\playwright-mcp-output\1765162676014")
OUTPUT_DIR = Path(__file__).parent
OUTPUT_PDF = OUTPUT_DIR / "SiteGiant_Import_Guide.pdf"

# Colors
COLOR_PRIMARY = (0, 242, 234)
COLOR_SECONDARY = (30, 30, 40)
COLOR_TEXT = (60, 60, 60)
COLOR_RED = (255, 107, 107)
COLOR_TEAL = (78, 205, 196)


def draw_box(draw, bbox, color="#FF6B6B", width=3):
    x1, y1, x2, y2 = bbox
    draw.rounded_rectangle([x1, y1, x2, y2], radius=8, outline=color, width=width)


def draw_callout(draw, position, number, color="#FF6B6B"):
    x, y = position
    r = 14
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x - tw // 2, y - th // 2 - 2), text, fill="white", font=font)


def annotate_webstore_listing(img_path, output_path):
    """Annotate the webstore listing page - showing how to access the menu"""
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img, "RGBA")
    # 1. Three-dots menu button (top right area near Add Product)
    draw_box(draw, (755, 78, 790, 102), "#FF6B6B", 3)
    draw_callout(draw, (772, 55), 1)
    img.convert("RGB").save(output_path, quality=95)


def annotate_import_page(img_path, output_path):
    """Annotate the import products page"""
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img, "RGBA")
    # 1. Sample template download link
    draw_box(draw, (115, 148, 265, 170), "#4ECDC4", 3)
    draw_callout(draw, (90, 159), 1, "#4ECDC4")
    # 2. File upload area
    draw_box(draw, (55, 185, 920, 430), "#FF6B6B", 3)
    draw_callout(draw, (30, 307), 2)
    # 3. Next button
    draw_box(draw, (875, 70, 925, 100), "#FF6B6B", 3)
    draw_callout(draw, (900, 50), 3)
    img.convert("RGB").save(output_path, quality=95)


class ImportGuidePDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*COLOR_SECONDARY)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 6)
        self.cell(0, 6, "SiteGiant Import Guide", align="L")
        self.set_xy(-30, 6)
        self.cell(20, 6, f"Page {self.page_no()}", align="R")

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "SiteGiant Pricing Automation", align="C")


def main():
    pdf = ImportGuidePDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # === Cover Page ===
    pdf.add_page()
    pdf.set_fill_color(*COLOR_SECONDARY)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(*COLOR_PRIMARY)
    pdf.set_xy(0, 80)
    pdf.cell(0, 15, "SiteGiant", align="C")

    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(0, 100)
    pdf.cell(0, 12, "Batch Price Import Guide", align="C")

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(180, 180, 180)
    pdf.set_xy(0, 125)
    pdf.cell(0, 8, "How to upload updated prices to SiteGiant", align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(0, 260)
    pdf.cell(0, 5, "SiteGiant Pricing Automation", align="C")

    # === Page 1: Overview ===
    pdf.add_page()
    pdf.set_text_color(*COLOR_TEXT)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Overview: Batch Price Update Workflow", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "After processing prices in the SiteGiant Pricing Automation tool, "
        "you'll have an Excel file ready to upload back to SiteGiant. "
        "This guide shows you how to import that file to update your product prices.",
    )
    pdf.ln(8)

    # Workflow steps
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "The Complete Workflow:", ln=True)

    steps = [
        ("1. Export from SiteGiant", "Download your current product list from SiteGiant"),
        ("2. Process in Pricing Tool", "Upload to our tool, review prices, export updated file"),
        ("3. Import back to SiteGiant", "Upload the updated Excel file (this guide)"),
    ]

    pdf.set_font("Helvetica", "", 11)
    for title, desc in steps:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"    {desc}", ln=True)
        pdf.set_text_color(*COLOR_TEXT)

    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*COLOR_TEAL[:3])
    pdf.cell(0, 8, "This guide focuses on Step 3: Importing to SiteGiant", ln=True)

    # === Page 2: Step 1 - Navigate to Webstore Listing ===
    pdf.add_page()
    pdf.set_text_color(*COLOR_TEXT)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Step 1: Navigate to Webstore Listing", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "1. Log in to your SiteGiant account at sitegiant.co\n"
        "2. In the left sidebar, click 'Products'\n"
        "3. Select 'Webstore Listing' from the submenu",
    )
    pdf.ln(5)

    # Add annotated screenshot if available
    webstore_img = SCREENSHOT_DIR / "sitegiant_webstore_listing.png"
    annotated_webstore = OUTPUT_DIR / "annotated_sg_webstore.png"
    if webstore_img.exists():
        annotate_webstore_listing(webstore_img, annotated_webstore)
        pdf.image(str(annotated_webstore), x=10, w=190)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*COLOR_RED[:3])
    pdf.cell(0, 6, "1  Click the three-dots menu (...) next to 'Add Product'", ln=True)

    # === Page 3: Step 2 - Open Import Menu ===
    pdf.add_page()
    pdf.set_text_color(*COLOR_TEXT)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Step 2: Select 'Import Products'", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "After clicking the three-dots menu, you'll see a dropdown with several options:\n\n"
        "  - Bulk Copy Listing\n"
        "  - Import Products  <-- Click this\n"
        "  - Export Product\n"
        "  - Exported / Imported\n"
        "  - Brands, Collections, etc.",
    )
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*COLOR_TEAL[:3])
    pdf.cell(0, 8, "Click 'Import Products' to open the import page.", ln=True)

    # === Page 4: Step 3 - Upload Excel File ===
    pdf.add_page()
    pdf.set_text_color(*COLOR_TEXT)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Step 3: Upload Your Excel File", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "On the Import Products page, you can upload your updated Excel file:")
    pdf.ln(3)

    # Add annotated screenshot
    import_img = SCREENSHOT_DIR / "sitegiant_import_products.png"
    annotated_import = OUTPUT_DIR / "annotated_sg_import.png"
    if import_img.exists():
        annotate_import_page(import_img, annotated_import)
        pdf.image(str(annotated_import), x=10, w=190)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*COLOR_TEAL[:3])
    pdf.cell(0, 6, "1  (Optional) Download sample template to check format", ln=True)
    pdf.set_text_color(*COLOR_RED[:3])
    pdf.cell(0, 6, "2  Drag your exported Excel file or click to browse", ln=True)
    pdf.cell(0, 6, "3  Click 'Next' to proceed with the import", ln=True)

    # === Page 5: Important Notes ===
    pdf.add_page()
    pdf.set_text_color(*COLOR_TEXT)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Important Notes", ln=True)

    pdf.set_font("Helvetica", "", 11)

    notes = [
        (
            "File Format",
            "Use the Excel file exported from our Pricing Automation tool. "
            "It's already formatted for SiteGiant import.",
        ),
        (
            "SKU Matching",
            "SiteGiant uses SKU to match products. Make sure your SKUs "
            "in the import file match your existing products.",
        ),
        ("Price Column", "The 'Selling Price' column contains your updated prices."),
        (
            "Review Before Confirm",
            "SiteGiant will show you a preview of changes. "
            "Review carefully before confirming the import.",
        ),
        (
            "Backup First",
            "Consider exporting your current prices before importing "
            "new ones, in case you need to revert.",
        ),
    ]

    for title, desc in notes:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*COLOR_SECONDARY)
        pdf.cell(0, 8, f"- {title}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 5, f"  {desc}")
        pdf.ln(3)

    # === Save PDF ===
    pdf.output(str(OUTPUT_PDF))
    print(f"Generated: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
