"""
Generate User Guide PDF with annotated screenshots - Professional Design v2.
"""

import os
from pathlib import Path

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

SCREENSHOT_DIR = Path(r"C:\Users\User\AppData\Local\Temp\playwright-mcp-output\1765162676014")
OUTPUT_DIR = Path(__file__).parent
OUTPUT_PDF = OUTPUT_DIR / "SiteGiant_Pricing_User_Guide.pdf"

COLOR_PRIMARY = (0, 242, 234)
COLOR_SECONDARY = (30, 30, 40)
COLOR_TEXT = (60, 60, 60)
COLOR_RED = (255, 107, 107)
COLOR_TEAL = (78, 205, 196)
COLOR_YELLOW = (255, 230, 109)
COLOR_GREEN = (149, 225, 211)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_font(size=16, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_callout(draw, position, number, color="#FF6B6B"):
    x, y = position
    radius = 16
    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)], fill=color, outline="white", width=2
    )
    font = get_font(18, bold=True)
    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x - tw / 2, y - th / 2 - 2), text, fill="white", font=font)


def draw_box(draw, bbox, color="#FF6B6B", width=3):
    x1, y1, x2, y2 = bbox
    draw.rounded_rectangle([x1, y1, x2, y2], radius=8, outline=color, width=width)


def annotate_home(img_path, output_path):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img, "RGBA")
    # Image is 1280x800 - scale coordinates accordingly
    # 1. Upload SiteGiant Export card (header + dashed drop zone)
    draw_box(draw, (302, 185, 838, 525), "#FF6B6B", 3)
    draw_callout(draw, (265, 355), 1)
    # 2. Process Prices button (teal button below upload area)
    draw_box(draw, (328, 568, 812, 622), "#FF6B6B", 3)
    draw_callout(draw, (290, 595), 2)
    # 3. System Status card (Pokedata API + Mapping File)
    draw_box(draw, (868, 218, 1218, 375), "#4ECDC4", 3)
    draw_callout(draw, (1235, 296), 3, "#4ECDC4")
    # 4. FX Rate & Margin card
    draw_box(draw, (868, 398, 1218, 508), "#4ECDC4", 3)
    draw_callout(draw, (1235, 453), 4, "#4ECDC4")
    img.convert("RGB").save(output_path, quality=95)


def annotate_mapping(img_path, output_path):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img, "RGBA")
    # Image is 1280x800 - scale coordinates accordingly
    # 1. Add button (green + Add)
    draw_box(draw, (838, 195, 900, 228), "#FF6B6B", 3)
    draw_callout(draw, (869, 170), 1)
    # 2. Upload button
    draw_box(draw, (912, 195, 998, 228), "#FF6B6B", 3)
    draw_callout(draw, (955, 170), 2)
    # 3. Refresh IDs button
    draw_box(draw, (1008, 195, 1115, 228), "#FF6B6B", 3)
    draw_callout(draw, (1062, 170), 3)
    # 4. Search bar
    draw_box(draw, (318, 262, 880, 310), "#4ECDC4", 3)
    draw_callout(draw, (280, 286), 4, "#4ECDC4")
    # 5. Table area
    draw_box(draw, (318, 325, 1210, 772), "#4ECDC4", 2)
    draw_callout(draw, (280, 548), 5, "#4ECDC4")
    img.convert("RGB").save(output_path, quality=95)


def annotate_config(img_path, output_path):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img, "RGBA")
    # Image is 1280x800 - scale coordinates accordingly
    # 1. API Key card
    draw_box(draw, (302, 187, 750, 393), "#FF6B6B", 3)
    draw_callout(draw, (265, 290), 1)
    # 2. Pricing card
    draw_box(draw, (778, 187, 1222, 393), "#4ECDC4", 3)
    draw_callout(draw, (1240, 290), 2, "#4ECDC4")
    # 3. Thresholds card
    draw_box(draw, (302, 435, 750, 698), "#FFE66D", 3)
    draw_callout(draw, (265, 566), 3, "#FFE66D")
    # 4. Mapping File card
    draw_box(draw, (778, 435, 1222, 698), "#95E1D3", 3)
    draw_callout(draw, (1240, 566), 4, "#95E1D3")
    img.convert("RGB").save(output_path, quality=95)


def annotate_explorer(img_path, output_path):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img, "RGBA")
    # Image is 1280x800 - scale coordinates accordingly
    # 1. Search box (text input with "Booster Box")
    draw_box(draw, (422, 106, 848, 143), "#FF6B6B", 3)
    draw_callout(draw, (385, 124), 1)
    # 2. Name/ID toggle buttons
    draw_box(draw, (858, 106, 978, 143), "#4ECDC4", 3)
    draw_callout(draw, (918, 75), 2, "#4ECDC4")
    # 3. Search button (teal button on right)
    draw_box(draw, (993, 106, 1108, 143), "#FF6B6B", 3)
    draw_callout(draw, (1130, 124), 3)
    # 4. Results table
    draw_box(draw, (298, 206, 1218, 778), "#4ECDC4", 2)
    draw_callout(draw, (260, 492), 4, "#4ECDC4")
    img.convert("RGB").save(output_path, quality=95)


def annotate_sg_webstore_cropped(img_path, output_path):
    """Crop and annotate the SiteGiant webstore listing - focus on header area"""
    img = Image.open(img_path)
    # Crop to show header + menu area (top portion with context)
    # Keep sidebar visible for navigation context
    cropped = img.crop((0, 50, 960, 450))  # x1, y1, x2, y2
    draw = ImageDraw.Draw(cropped, "RGBA")
    # 1. Three-dots menu button (adjusted for crop)
    draw_box(draw, (755, 28, 790, 52), "#FF6B6B", 3)
    draw_callout(draw, (772, 8), 1)
    # 2. Products menu in sidebar
    draw_box(draw, (18, 118, 195, 145), "#4ECDC4", 2)
    draw_callout(draw, (208, 131), 2, "#4ECDC4")
    cropped.convert("RGB").save(output_path, quality=95)


def annotate_sg_import_cropped(img_path, output_path):
    """Crop and annotate the SiteGiant import page - focus on upload area"""
    img = Image.open(img_path)
    # Crop to show import area (remove footer)
    cropped = img.crop((0, 0, 960, 480))  # x1, y1, x2, y2
    draw = ImageDraw.Draw(cropped, "RGBA")
    # 1. Sample template download link
    draw_box(draw, (115, 148, 265, 170), "#4ECDC4", 3)
    draw_callout(draw, (90, 159), 1, "#4ECDC4")
    # 2. File upload area
    draw_box(draw, (55, 185, 920, 430), "#FF6B6B", 3)
    draw_callout(draw, (30, 307), 2)
    # 3. Next button
    draw_box(draw, (875, 70, 925, 100), "#FF6B6B", 3)
    draw_callout(draw, (900, 50), 3)
    cropped.convert("RGB").save(output_path, quality=95)


class UserGuidePDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*COLOR_SECONDARY)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 5)
        self.cell(95, 8, "SiteGiant Pricing Automation")
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(95, 8, "User Guide", align="R")
        self.ln(18)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*COLOR_SECONDARY)
        x, y = self.get_x(), self.get_y()
        self.set_fill_color(*COLOR_PRIMARY)
        self.rect(x, y + 3, 4, 7, "F")
        self.set_x(x + 8)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*COLOR_TEXT)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def add_image(self, path, caption=None):
        if not os.path.exists(path):
            return
        self.ln(2)
        w = self.w - 2 * self.l_margin
        self.image(str(path), x=self.l_margin, w=w)
        if caption:
            self.ln(1)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 4, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def step(self, num, title, desc, color=COLOR_RED):
        y = self.get_y()
        self.set_fill_color(*color)
        self.ellipse(self.l_margin, y + 1, 6, 6, style="F")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_xy(self.l_margin, y + 1)
        self.cell(6, 6, str(num), align="C")
        self.set_xy(self.l_margin + 9, y)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.cell(0, 5, title, new_x="LMARGIN", new_y="NEXT")
        if desc:
            self.set_x(self.l_margin + 9)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 4, desc)
        self.ln(2)


def generate_pdf():
    pdf = UserGuidePDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover
    pdf.add_page()
    pdf.set_fill_color(*COLOR_SECONDARY)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_fill_color(*COLOR_PRIMARY)
    pdf.rect(0, 0, 210, 12, "F")
    pdf.rect(0, 285, 210, 12, "F")
    pdf.ln(70)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 32)
    pdf.cell(0, 12, "SiteGiant Pricing", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*COLOR_PRIMARY)
    pdf.set_font("Helvetica", "B", 28)
    pdf.cell(0, 12, "Automation Tool", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(*COLOR_PRIMARY)
    pdf.line(75, pdf.get_y(), 135, pdf.get_y())
    pdf.ln(8)
    pdf.set_text_color(180, 180, 180)
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 10, "USER GUIDE", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(70)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "Version 1.0 | December 2024", align="C")

    # TOC
    pdf.add_page()
    pdf.section_title("Contents")
    pdf.ln(5)
    for title, pg in [
        ("1. Quick Start", 3),
        ("2. Home - Price Update", 4),
        ("3. Mapping Manager", 5),
        ("4. Configuration", 6),
        ("5. Explorer", 7),
        ("6. Upload to SiteGiant", 8),
    ]:
        pdf.set_text_color(*COLOR_SECONDARY)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(160, 8, title)
        pdf.set_text_color(*COLOR_PRIMARY)
        pdf.cell(20, 8, str(pg), align="R", new_x="LMARGIN", new_y="NEXT")

    # 1. Quick Start
    pdf.add_page()
    pdf.section_title("1. Quick Start")
    pdf.body_text(
        "This tool syncs SiteGiant prices with Pokedata market values. Follow these steps:"
    )
    pdf.ln(3)
    pdf.step(1, "Set API Key", "Go to Settings > Config and enter your Pokedata API key.")
    pdf.step(2, "Create Mapping", "Link your SiteGiant SKUs to Pokedata products.")
    pdf.step(3, "Upload Export", "Upload your SiteGiant product export file on Home page.")
    pdf.step(4, "Process", "Click 'Process Prices' to fetch prices and generate output.")

    # 2. Home
    pdf.add_page()
    pdf.section_title("2. Home - Price Update")
    pdf.body_text("The main dashboard for uploading exports and processing prices.")
    home_img = SCREENSHOT_DIR / "guide_01_home.png"
    if home_img.exists():
        annotated = OUTPUT_DIR / "annotated_home.png"
        annotate_home(home_img, annotated)
        pdf.add_image(annotated)
    pdf.step(1, "Upload Area", "Drag & drop your SiteGiant .xlsx export file here.")
    pdf.step(2, "Process Button", "Click to fetch Pokedata prices and generate output.")
    pdf.step(3, "System Status", "Shows API connection and mapping file status.", COLOR_TEAL)
    pdf.step(4, "FX Rate & Margin", "Current exchange rate and margin settings.", COLOR_TEAL)

    # 3. Mapping
    pdf.add_page()
    pdf.section_title("3. Mapping Manager")
    pdf.body_text("Link your SKUs to Pokedata products. Auto-lookup finds IDs from URLs.")
    mapping_img = SCREENSHOT_DIR / "guide_03_mapping_table.png"
    if mapping_img.exists():
        annotated = OUTPUT_DIR / "annotated_mapping.png"
        annotate_mapping(mapping_img, annotated)
        pdf.add_image(annotated)
    pdf.step(1, "Add", "Manually add a single SKU mapping.")
    pdf.step(2, "Upload", "Bulk import mappings from Excel/CSV.")
    pdf.step(3, "Refresh IDs", "Auto-lookup missing Pokedata IDs from URLs.")
    pdf.step(4, "Search", "Filter mappings by SKU, name, or URL.", COLOR_TEAL)
    pdf.step(5, "Table", "View and edit mappings. Click icons for actions.", COLOR_TEAL)

    # 4. Config
    pdf.add_page()
    pdf.section_title("4. Configuration")
    pdf.body_text("Set your API key, pricing parameters, and thresholds.")
    config_img = SCREENSHOT_DIR / "guide_04_config.png"
    if config_img.exists():
        annotated = OUTPUT_DIR / "annotated_config.png"
        annotate_config(config_img, annotated)
        pdf.add_image(annotated)
    pdf.step(1, "API Key", "Your Pokedata API key. Required for pricing data.")
    pdf.step(2, "Pricing", "FX Rate (USD to MYR) and Margin Divisor.", COLOR_TEAL)
    pdf.step(3, "Thresholds", "Warning and Block limits for price changes.", COLOR_YELLOW)
    pdf.step(4, "Mapping File", "Download or replace the mapping file.", COLOR_GREEN)

    # 5. Explorer
    pdf.add_page()
    pdf.section_title("5. Explorer")
    pdf.body_text("Search Pokedata products to find IDs for your mapping.")
    explorer_img = SCREENSHOT_DIR / "guide_07_explorer_results.png"
    if explorer_img.exists():
        annotated = OUTPUT_DIR / "annotated_explorer.png"
        annotate_explorer(explorer_img, annotated)
        pdf.add_image(annotated)
    pdf.step(1, "Search Box", "Enter a product name to search.")
    pdf.step(2, "Mode Toggle", "Switch between Name and ID search.", COLOR_TEAL)
    pdf.step(3, "Search Button", "Click to execute search.")
    pdf.step(4, "Results", "Click 'View Pricing' to see price history.", COLOR_TEAL)

    # 6. Upload to SiteGiant
    pdf.add_page()
    pdf.section_title("6. Upload to SiteGiant")
    pdf.body_text("After exporting updated prices, upload them back to SiteGiant.")
    pdf.ln(2)

    # Step 1: Navigate
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.cell(0, 6, "Step 1: Navigate to Webstore Listing", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*COLOR_TEXT)
    pdf.multi_cell(0, 5, "Log in to sitegiant.co > Products > Webstore Listing")
    pdf.ln(2)

    sg_webstore = SCREENSHOT_DIR / "sitegiant_webstore_listing.png"
    if sg_webstore.exists():
        annotated = OUTPUT_DIR / "annotated_sg_webstore_crop.png"
        annotate_sg_webstore_cropped(sg_webstore, annotated)
        pdf.add_image(annotated)
    pdf.step(1, "Three-Dots Menu", "Click to open the dropdown menu.")
    pdf.step(2, "Products Menu", "Shows you're in the right section.", COLOR_TEAL)
    pdf.ln(2)

    # Step 2: Select Import
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.cell(0, 6, "Step 2: Select Import Products", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*COLOR_TEXT)
    pdf.multi_cell(0, 5, "From the dropdown menu, click 'Import Products'.")
    pdf.ln(3)

    # Step 3: Upload file
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.cell(0, 6, "Step 3: Upload Excel File", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    sg_import = SCREENSHOT_DIR / "sitegiant_import_products.png"
    if sg_import.exists():
        annotated = OUTPUT_DIR / "annotated_sg_import_crop.png"
        annotate_sg_import_cropped(sg_import, annotated)
        pdf.add_image(annotated)
    pdf.step(1, "Sample Template", "Optional: Download to check format.", COLOR_TEAL)
    pdf.step(2, "Upload Area", "Drag your exported Excel file here.")
    pdf.step(3, "Next Button", "Click to proceed with import.")

    pdf.output(str(OUTPUT_PDF))
    print(f"Generated: {OUTPUT_PDF}")


if __name__ == "__main__":
    generate_pdf()
