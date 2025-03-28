# main.py (Backend modifications)
from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os
import uuid
import re
from datetime import datetime, timedelta
import io
from bs4 import BeautifulSoup
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from PIL import Image  # For image processing

app = FastAPI()

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

def cleanup_old_files(directory, hours=24):
    now = datetime.now()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_time > timedelta(hours=hours):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")

@app.on_event("startup")
async def startup_event():
    cleanup_old_files(OUTPUT_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Template error: {str(e)}")

@app.post("/preview_report")
async def preview_report(
    request: Request,
    eventType: str = Form(...),
    department: str = Form(...),
    topic: str = Form(...),
    expertName: Optional[str] = Form(None),
    venue: str = Form(...),
    eventDurationType: str = Form(...),
    date: Optional[str] = Form(None),
    startTime: Optional[str] = Form(None),
    endTime: Optional[str] = Form(None),
    startDate: Optional[str] = Form(None),
    endDate: Optional[str] = Form(None),
    coordinator: str = Form(...),
    participants: int = Form(...),
    summary: str = Form(...),
    outcome: str = Form(...),
    hodName: str = Form(...)
):
    try:
        # Format date/time consistently for both preview and download
        if eventDurationType.lower() == "multiple" and startDate and endDate:
            formatted_dateTime = (
                datetime.strptime(startDate, "%Y-%m-%d").strftime("%B %d, %Y")
                + " to " +
                datetime.strptime(endDate, "%Y-%m-%d").strftime("%B %d, %Y")
            )
        elif date and startTime and endTime:
            formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y")
            formatted_time = (
                datetime.strptime(startTime, "%H:%M").strftime("%I:%M %p")
                + " - " +
                datetime.strptime(endTime, "%H:%M").strftime("%I:%M %p")
            )
            formatted_dateTime = formatted_date + ", " + formatted_time
        else:
            # Fallback in case of missing data
            formatted_dateTime = "Date information not provided"

        report_data = {
            "eventType": eventType.title(),
            "department": department,
            "topic": topic,
            "expertName": "N/A" if eventType.lower() == "field visit" else (expertName if expertName else "N/A"),
            "venue": venue,
            "dateTime": formatted_dateTime,
            "coordinator": coordinator,
            "participants": participants,
            "summary": summary,
            "outcome": outcome,
            "hodName": hodName
        }
        
        # Add query parameters for image counts to be used by JavaScript
        response = templates.TemplateResponse("preview.html", {"request": request, "report": report_data})
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")

def replace_placeholder(doc, placeholder, value):
    def process_paragraph(paragraph, ph, val):
        if ph in paragraph.text:
            full_text = ''.join(run.text for run in paragraph.runs)
            if ph in full_text:
                paragraph.clear()
                new_run = paragraph.add_run(full_text.replace(ph, str(val)))
                new_run.font.name = 'DIN Pro Regular'
                # Increase font size for event type and department
                if placeholder == "{{eventType}}" or placeholder == "{{department}}":
                    new_run.font.size = Pt(14)
                else:
                    new_run.font.size = Pt(11)
                # Set color to blue for specific text
                if paragraph.text.strip().endswith(" Report") or paragraph.text.strip().startswith("Department of ") or placeholder == "{{eventType}}" or placeholder == "{{department}}":
                    new_run.font.color.rgb = RGBColor(0, 112, 192)  # Blue color
    for paragraph in doc.paragraphs:
        process_paragraph(paragraph, placeholder, value)
    def process_table(table):
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    process_paragraph(paragraph, placeholder, value)
                for nested_table in cell.tables:
                    process_table(nested_table)
    for table in doc.tables:
        process_table(table)

def process_node_formatting(paragraph, node):
    if isinstance(node, str):
        run = paragraph.add_run(node)
        run.font.name = 'DIN Pro Regular'
        run.font.size = Pt(11)
        return run
    if not hasattr(node, 'name'):
        run = paragraph.add_run(str(node))
        run.font.name = 'DIN Pro Regular'
        run.font.size = Pt(11)
        return run
    if node.name in ['strong', 'b']:
        last_run = None
        if node.contents:
            for child in node.contents:
                last_run = process_node_formatting(paragraph, child)
                if last_run is not None:
                    last_run.bold = True
        else:
            last_run = paragraph.add_run(node.get_text())
            last_run.bold = True
            last_run.font.name = 'DIN Pro Regular'
            last_run.font.size = Pt(11)
        return last_run
    elif node.name in ['em', 'i']:
        last_run = None
        if node.contents:
            for child in node.contents:
                last_run = process_node_formatting(paragraph, child)
                if last_run is not None:
                    last_run.italic = True
        else:
            last_run = paragraph.add_run(node.get_text())
            last_run.italic = True
            last_run.font.name = 'DIN Pro Regular'
            last_run.font.size = Pt(11)
        return last_run
    elif node.contents:
        last_run = None
        for child in node.contents:
            last_run = process_node_formatting(paragraph, child)
        return last_run
    else:
        run = paragraph.add_run(node.get_text())
        run.font.name = 'DIN Pro Regular'
        run.font.size = Pt(11)
        return run

def insert_paragraph_after(paragraph, text='', style=None):
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if text:
        new_para.add_run(text)
    if style:
        new_para.style = style
    return new_para

def replace_placeholder_with_html(doc, placeholder, html_content):
    paragraphs = list(doc.paragraphs)
    for i, paragraph in enumerate(paragraphs):
        if placeholder in paragraph.text:
            paragraph.clear()
            soup = BeautifulSoup(html_content, "html.parser")
            last_par = paragraph
            elements = list(soup.body.children) if soup.body else list(soup.children)
            if not elements:
                last_par.add_run(soup.get_text())
                return
            first = True
            for element in elements:
                if isinstance(element, str) and element.strip() == '':
                    continue
                if element.name == 'p':
                    if first:
                        new_par = last_par
                        first = False
                    else:
                        new_par = insert_paragraph_after(last_par)
                    for child in element.children:
                        process_node_formatting(new_par, child)
                    last_par = new_par
                elif element.name in ['ul', 'ol']:
                    for li in element.find_all('li', recursive=False):
                        list_style = 'List Bullet' if element.name == 'ul' else 'List Number'
                        new_par = insert_paragraph_after(last_par, style=list_style)
                        for child in li.children:
                            process_node_formatting(new_par, child)
                        last_par = new_par
                elif isinstance(element, str) and element.strip():
                    new_par = last_par if first else insert_paragraph_after(last_par)
                    run = new_par.add_run(element.strip())
                    run.font.name = 'DIN Pro Regular'
                    run.font.size = Pt(11)
                    last_par = new_par
            break

def update_header(doc, eventType):
    for paragraph in doc.paragraphs:
        if "{{eventType}}" in paragraph.text:
            text = paragraph.text.replace("{{eventType}}", eventType.title())
            paragraph.clear()
            run = paragraph.add_run(text)
            run.font.name = 'DIN Pro Regular'
            run.font.size = Pt(24)
            run.bold = True
            run.font.color.rgb = RGBColor(0, 112, 192)  # Blue
            break

def set_section_vertical_alignment_bottom(section):
    try:
        sectPr = section._sectPr
        for child in sectPr.findall(qn('w:valign')):
            sectPr.remove(child)
        vAlign = OxmlElement('w:valign')
        vAlign.set(qn('w:val'), 'bottom')
        sectPr.append(vAlign)
    except Exception as e:
        print(f"Could not set vertical alignment: {e}")

def remove_table_borders(table):
    tbl = table._tbl  # Get the underlying XML table element
    tblPr = tbl.find(qn('w:tblPr'))  # Find table properties
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    
    # Remove existing table borders
    tblBorders = tblPr.find(qn('w:tblBorders'))
    if tblBorders is not None:
        tblPr.remove(tblBorders)
    
    # Create new table borders element with no borders
    new_tblBorders = OxmlElement('w:tblBorders')
    new_tblBorders.set(qn('w:top'), "none")
    new_tblBorders.set(qn('w:start'), "none")
    new_tblBorders.set(qn('w:end'), "none")
    new_tblBorders.set(qn('w:bottom'), "none")
    new_tblBorders.set(qn('w:insideH'), "none")
    new_tblBorders.set(qn('w:insideV'), "none")
    
    tblPr.append(new_tblBorders)

def add_signature_section(doc, coordinator, hodName):
    # Create a table for signatures
    table = doc.add_table(rows=1, cols=2)
    table.autofit = False
    table.columns[0].width = Inches(3.5)
    table.columns[1].width = Inches(3.5)
    
    # Remove all borders from the table
    tbl = table._tbl  # Get the underlying XML table element
    tblPr = tbl.find(qn('w:tblPr'))  # Find table properties
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    
    # Remove existing table borders
    tblBorders = tblPr.find(qn('w:tblBorders'))
    if tblBorders is not None:
        tblPr.remove(tblBorders)
    
    # Create new table borders element with no borders
    new_tblBorders = OxmlElement('w:tblBorders')
    new_tblBorders.set(qn('w:top'), "none")
    new_tblBorders.set(qn('w:start'), "none")
    new_tblBorders.set(qn('w:end'), "none")
    new_tblBorders.set(qn('w:bottom'), "none")
    new_tblBorders.set(qn('w:insideH'), "none")
    new_tblBorders.set(qn('w:insideV'), "none")
    
    tblPr.append(new_tblBorders)

    # Add signature text
    cell = table.cell(0, 0)
    cell.text =  "\n\n\n\n Name & Signature of Faculty-in-charge\n" + coordinator
    cell = table.cell(0, 1)
    cell.text = "\n\n\n\n Name & Signature of HoD\n" + hodName
    
    # Format the table 
    for i, cell in enumerate(table.rows[0].cells):
        for paragraph in cell.paragraphs:
            # Left align for coordinator (first cell)
            if i == 0:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            # Right align for HOD (second cell)
            else:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            for run in paragraph.runs:
                run.font.name = 'DIN Pro Regular'
                run.font.size = Pt(11)
@app.post("/generate_report")
async def generate_report(
    eventType: str = Form(...),
    department: str = Form(...),
    topic: str = Form(...),
    expertName: Optional[str] = Form(None),
    venue: str = Form(...),
    eventDurationType: str = Form(...),
    date: Optional[str] = Form(None),
    startTime: Optional[str] = Form(None),
    endTime: Optional[str] = Form(None),
    startDate: Optional[str] = Form(None),
    endDate: Optional[str] = Form(None),
    coordinator: str = Form(...),
    participants: int = Form(...),
    summary: str = Form(...),
    outcome: str = Form(...),
    hodName: str = Form(...),
    invitePoster: Optional[List[UploadFile]] = File([]),
    actionPhotos: Optional[List[UploadFile]] = File([]),
    attendanceSheet: Optional[List[UploadFile]] = File([]),
    analysisReport: Optional[List[UploadFile]] = File([])
):
    try:
        if eventDurationType.lower() == "multiple" and startDate and endDate:
            formatted_dateTime = (
                datetime.strptime(startDate, "%Y-%m-%d").strftime("%B %d, %Y")
                + " to " +
                datetime.strptime(endDate, "%Y-%m-%d").strftime("%B %d, %Y")
            )
        elif date and startTime and endTime:
            formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y")
            formatted_time = (
                datetime.strptime(startTime, "%H:%M").strftime("%I:%M %p")
                + " - " +
                datetime.strptime(endTime, "%H:%M").strftime("%I:%M %p")
            )
            formatted_dateTime = formatted_date + ", " + formatted_time
        else:
            # Fallback in case of missing data
            formatted_dateTime = "Date information not provided"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = re.sub(r'[^\w\s]', '_', topic)
        output_filename = f"{department}_{safe_topic.replace(' ', '_')}_{timestamp}.docx"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        template_name = f"{eventType.lower()}_template.docx"
        template_path = os.path.join(TEMPLATES_DIR, template_name)
        if not os.path.exists(template_path):
            template_path = os.path.join(TEMPLATES_DIR, "workshop_template.docx")
            if not os.path.exists(template_path):
                doc = Document()
                doc.add_paragraph("{{eventType}} Report")
            else:
                doc = Document(template_path)
        else:
            doc = Document(template_path)

        replacements = {
            "{{eventType}}": eventType.title(),
            "{{department}}": department,
            "{{topic}}": topic,
            "{{expertName}}": expertName if expertName else "N/A",
            "{{venue}}": venue,
            "{{dateTime}}": formatted_dateTime,
            "{{coordinator}}": coordinator,
            "{{participants}}": str(participants),
            "{{hodName}}": hodName
        }
        for placeholder, value in replacements.items():
            replace_placeholder(doc, placeholder, value)
        
        # Remove expert name row for field visits
        if eventType.lower() == 'field visit':
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if '{{expertName}}' in cell.text:
                            table._tbl.remove(row._tr)
                            break
        replace_placeholder_with_html(doc, "{{summary}}", summary)
        replace_placeholder_with_html(doc, "{{outcome}}", outcome)
        # Add page break after outcome section
        doc.add_page_break()
        update_header(doc, eventType)

        image_sections = [
            ("Invite Poster", invitePoster),
            ("Action Photos", actionPhotos),
            ("Attendance Sheet", attendanceSheet),
            ("Analysis Report", analysisReport)
        ]
        for section_name, images in image_sections:
            if images:
                valid_images = [img for img in images if img and img.filename] if images else []
                if valid_images:
                    
                    # Add section title
                    p = doc.add_paragraph()
                    run = p.add_run(section_name)
                    run.bold = True
                    run.font.size = Pt(16)
                    run.font.name = 'DIN Pro Regular'
                    run.font.color.rgb = RGBColor(0, 112, 192)  # Blue color
                    
                    # Analyze images to determine optimal layout
                    # Count portrait and landscape images to determine best layout
                    portrait_count = 0
                    landscape_count = 0
                    square_count = 0
                    
                    # Temporarily save images to analyze their dimensions
                    temp_image_paths = []
                    for img in valid_images:
                        unique_filename = f"{uuid.uuid4()}_{img.filename}"
                        img_path = os.path.join(UPLOAD_DIR, unique_filename)
                        with open(img_path, "wb") as buffer:
                            await img.seek(0)
                            contents = await img.read()
                            buffer.write(contents)
                        
                        # Analyze image dimensions
                        try:
                            with Image.open(img_path) as pil_img:
                                img_width, img_height = pil_img.size
                                aspect_ratio = img_width / img_height
                                
                                if aspect_ratio < 0.8:  # Portrait
                                    portrait_count += 1
                                elif aspect_ratio > 1.2:  # Landscape
                                    landscape_count += 1
                                else:  # Square-ish
                                    square_count += 1
                                    
                                temp_image_paths.append((img_path, aspect_ratio))
                        except Exception as e:
                            print(f"Error analyzing image {img.filename}: {e}")
                            # If analysis fails, assume it's a standard image
                            temp_image_paths.append((img_path, 1.0))
                    
                    # Determine optimal layout based on image count and orientations
                    if len(valid_images) <= 2:
                        # For 1-2 images, use a single column if both are portrait
                        if portrait_count == len(valid_images):
                            cols = 1
                            rows = len(valid_images)
                        else:
                            # Otherwise use 2 columns for 2 images, 1 column for 1 image
                            cols = min(2, len(valid_images))
                            rows = (len(valid_images) + cols - 1) // cols  # Ceiling division
                    elif len(valid_images) <= 4:
                        # For 3-4 images, use 2 columns
                        cols = 2
                        rows = (len(valid_images) + 1) // 2
                    else:
                        # For 5+ images, use 3 columns if mostly portrait, otherwise 2 columns
                        if portrait_count > (len(valid_images) // 2):
                            cols = 3
                        else:
                            cols = 2
                        rows = (len(valid_images) + cols - 1) // cols  # Ceiling division
                    
                    # Create a table for the images with the determined layout
                    table = doc.add_table(rows=rows, cols=cols)
                    table.alignment = WD_TABLE_ALIGNMENT.CENTER
                    
                    # Remove borders from the table for clean layout
                    remove_table_borders(table)
                    
                    # Add spacing between cells for better image separation
                    tbl = table._tbl
                    tblPr = tbl.find(qn('w:tblPr'))
                    if tblPr is None:
                        tblPr = OxmlElement('w:tblPr')
                        tbl.insert(0, tblPr)
                    
                    # Set cell spacing to exactly 0.2 inches (288 twips) on all sides
                    tblCellMar = OxmlElement('w:tblCellMar')
                    
                    # Add spacing on all sides (left, right, top, bottom)
                    for side in ['top', 'start', 'bottom', 'end']:
                        spacing = OxmlElement(f'w:{side}')
                        spacing.set(qn('w:w'), '288')
                        spacing.set(qn('w:type'), 'dxa')
                        tblCellMar.append(spacing)
                    
                    tblPr.append(tblCellMar)
                    
                    # Add images to the table cells
                    img_index = 0
                    for row_idx in range(rows):
                        for col_idx in range(cols):
                            if img_index < len(valid_images):
                                img = valid_images[img_index]
                                cell = table.cell(row_idx, col_idx)
                                
                                # Add cell margins for better spacing between images
                                tc = cell._tc
                                tcPr = tc.get_or_add_tcPr()
                                tcMar = OxmlElement('w:tcMar')
                                
                                # Add margin on all sides (left, right, top, bottom) - exactly 0.2 inches (288 twips)
                                for side in ['top', 'start', 'bottom', 'end']:
                                    node = OxmlElement(f'w:{side}')
                                    node.set(qn('w:w'), '288')
                                    node.set(qn('w:type'), 'dxa')
                                    tcMar.append(node)
                                
                                tcPr.append(tcMar)
                                
                                # Center align the content in the cell
                                if cell.paragraphs:
                                    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                                
                                # Save the image temporarily
                                unique_filename = f"{uuid.uuid4()}_{img.filename}"
                                img_path = os.path.join(UPLOAD_DIR, unique_filename)
                                with open(img_path, "wb") as buffer:
                                    await img.seek(0)
                                    contents = await img.read()
                                    buffer.write(contents)
                                
                                try:
                                    # Get the saved image path and aspect ratio
                                    img_path, aspect_ratio = temp_image_paths[img_index]
                                    
                                    # Calculate optimal image size based on number of columns and aspect ratio
                                    # Base width depends on number of columns
                                    if cols == 1:
                                        base_width = Inches(6.0)  # Full width for single column
                                    elif cols == 2:
                                        base_width = Inches(3.0)  # Half width for 2 columns
                                    else:  # cols == 3
                                        base_width = Inches(2.0)  # Third width for 3 columns
                                    
                                    # Adjust width based on aspect ratio
                                    if aspect_ratio < 0.8:  # Portrait
                                        # For portrait images, reduce width to avoid excessive height
                                        img_width = base_width * 0.85
                                    elif aspect_ratio > 1.5:  # Wide landscape
                                        # For wide landscape, use full base width
                                        img_width = base_width
                                    else:  # Standard/square images
                                        img_width = base_width * 0.9
                                    
                                    # Reduce paragraph spacing for compact layout
                                    paragraph = cell.paragraphs[0]
                                    paragraph.paragraph_format.space_before = Pt(4)
                                    paragraph.paragraph_format.space_after = Pt(4)
                                    
                                    # Add the image with proper spacing and adaptive width
                                    run = paragraph.add_run()
                                    run.add_picture(img_path, width=img_width)
                                    
                                    # Clean up the temporary file
                                    os.remove(img_path)
                                except Exception as e:
                                    print(f"Error processing image {img.filename}: {e}")
                                
                                img_index += 1
                    
                    # Clean up any remaining temporary files
                    for img_path, _ in temp_image_paths:
                        if os.path.exists(img_path):
                            try:
                                os.remove(img_path)
                            except Exception as e:
                                print(f"Error removing temporary file {img_path}: {e}")
                    
                    # Add page break after each section, except for Analysis Report
                    # This ensures each image section starts on a new page
                    if section_name != "Analysis Report":
                        doc.add_page_break()

        # Add signature section
        add_signature_section(doc, coordinator, hodName)

        doc.save(output_path)
        cleanup_old_files(OUTPUT_DIR, hours=24)
        return {
            "message": "Report generated successfully",
            "filename": output_filename,
            "download_url": f"/download_report/{output_filename}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to generate report: {str(e)}"}

@app.get("/download_report/{filename}")
async def download_report(filename: str):
    sanitized_filename = os.path.basename(filename)
    if sanitized_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join(OUTPUT_DIR, sanitized_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=sanitized_filename
    )

@app.on_event("startup")
async def create_default_template():
    default_template_path = os.path.join(TEMPLATES_DIR, "workshop_template.docx")
    if not os.path.exists(default_template_path):
        try:
            doc = Document()
            p_title = doc.add_paragraph()
            run_title = p_title.add_run("{{eventType}} Report")
            run_title.font.name = 'DIN Pro Regular'
            run_title.font.size = Pt(16)  # Changed from 24 to 16
            run_title.bold = True
            run_title.font.color.rgb = RGBColor(0, 112, 192)  # Blue color

            p_details = doc.add_paragraph("Event Details")
            p_details.runs[0].font.name = 'DIN Pro Regular'
            p_details.runs[0].font.size = Pt(16)
            p_details.runs[0].bold = True
            p_details.runs[0].font.color.rgb = RGBColor(0, 112, 192)  # Blue color
            details = [
                ("Department", "{{department}}"),
                ("Topic", "{{topic}}"),
                ("Expert Name", "{{expertName}}"),
                ("Venue", "{{venue}}"),
                ("Event Date/Time", "{{dateTime}}"),
                ("Faculty Coordinator", "{{coordinator}}"),
                ("HOD Name", "{{hodName}}"),
                ("Participants", "{{participants}}")
            ]
            table = doc.add_table(rows=len(details), cols=2)
            table.style = 'Table Grid'
            for i, (label, value) in enumerate(details):
                row = table.rows[i]
                row.cells[0].text = label
                row.cells[1].text = value
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.name = 'DIN Pro Regular'
                            run.font.size = Pt(11)
            p_summary = doc.add_paragraph("Summary")
            p_summary.runs[0].font.name = 'DIN Pro Regular'
            p_summary.runs[0].font.size = Pt(16)
            p_outcome = doc.add_paragraph("Outcome")
            p_outcome.runs[0].font.name = 'DIN Pro Regular'
            p_outcome.runs[0].font.size = Pt(16)
            # Signature table is added dynamically in generate_report, not here
            doc.save(default_template_path)
            print(f"Created default template at {default_template_path}")
        except Exception as e:
            print(f"Failed to create default template: {e}")

@app.on_event("startup")
async def schedule_cleanup():
    import asyncio
    async def periodic_cleanup():
        while True:
            cleanup_old_files(OUTPUT_DIR)
            await asyncio.sleep(3600)
    asyncio.create_task(periodic_cleanup())

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use the PORT environment variable if set, otherwise default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
