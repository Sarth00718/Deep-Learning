#!/usr/bin/env python3
"""
Convert Jupyter notebooks to PDFs with custom header format
"""

import os
import glob
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
import json
import base64
from io import BytesIO
from PIL import Image as PILImage

# Student information
STUDENT_NAME = "NAROLA SARTH DHARMESHBHAI"
ROLL_NO = "23BCE194"
SUBJECT = "COMPUTER VISION USING DEEP LEARNING"

def create_header_page(story, practical_number, practical_name):
    """Create the header page with student information"""
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='black',
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    # Add header information
    story.append(Paragraph(f"NAME: {STUDENT_NAME}", info_style))
    story.append(Paragraph(f"ROLL NO: {ROLL_NO}", info_style))
    story.append(Paragraph(f"SUBJECT: {SUBJECT}", info_style))
    story.append(Paragraph(f"PRACTICAL NO. AND NAME: {practical_name}", info_style))
    story.append(Spacer(1, 0.5*inch))

def process_notebook_cell(cell, story, styles):
    """Process a single notebook cell"""
    cell_type = cell.get('cell_type', '')
    
    if cell_type == 'markdown':
        # Process markdown cell
        source = ''.join(cell.get('source', []))
        if source.strip():
            # Simple markdown to text conversion
            lines = source.split('\n')
            for line in lines:
                if line.startswith('#'):
                    # Header
                    level = len(line) - len(line.lstrip('#'))
                    text = line.lstrip('#').strip()
                    if level == 1:
                        story.append(Paragraph(text, styles['Heading1']))
                    elif level == 2:
                        story.append(Paragraph(text, styles['Heading2']))
                    else:
                        story.append(Paragraph(text, styles['Heading3']))
                elif line.strip():
                    story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
    
    elif cell_type == 'code':
        # Process code cell
        source = ''.join(cell.get('source', []))
        if source.strip():
            story.append(Paragraph("<b>CODE:</b>", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            # Add code with monospace font
            code_style = ParagraphStyle(
                'Code',
                parent=styles['Code'],
                fontSize=8,
                leftIndent=20,
                fontName='Courier'
            )
            
            # Split long lines
            lines = source.split('\n')
            for line in lines:
                if len(line) > 80:
                    # Wrap long lines
                    for i in range(0, len(line), 80):
                        story.append(Preformatted(line[i:i+80], code_style))
                else:
                    story.append(Preformatted(line, code_style))
            
            story.append(Spacer(1, 0.2*inch))
            
            # Process outputs
            outputs = cell.get('outputs', [])
            if outputs:
                story.append(Paragraph("<b>OUTPUT:</b>", styles['Heading3']))
                story.append(Spacer(1, 0.1*inch))
                
                for output in outputs:
                    output_type = output.get('output_type', '')
                    
                    if output_type == 'stream':
                        text = ''.join(output.get('text', []))
                        if text.strip():
                            story.append(Preformatted(text[:500], code_style))  # Limit output length
                    
                    elif output_type == 'execute_result' or output_type == 'display_data':
                        # Handle text output
                        if 'text/plain' in output.get('data', {}):
                            text = ''.join(output['data']['text/plain'])
                            story.append(Preformatted(text[:500], code_style))
                        
                        # Handle image output
                        if 'image/png' in output.get('data', {}):
                            try:
                                img_data = output['data']['image/png']
                                img_bytes = base64.b64decode(img_data)
                                img = PILImage.open(BytesIO(img_bytes))
                                
                                # Resize image if too large
                                max_width = 500
                                max_height = 400
                                img.thumbnail((max_width, max_height), PILImage.Resampling.LANCZOS)
                                
                                # Save to temporary buffer
                                img_buffer = BytesIO()
                                img.save(img_buffer, format='PNG')
                                img_buffer.seek(0)
                                
                                # Add to PDF
                                from reportlab.platypus import Image as RLImage
                                rl_img = RLImage(img_buffer, width=img.width, height=img.height)
                                story.append(rl_img)
                            except Exception as e:
                                story.append(Paragraph(f"[Image output - could not render: {str(e)}]", styles['Normal']))
                    
                    elif output_type == 'error':
                        error_text = '\n'.join(output.get('traceback', []))
                        story.append(Preformatted(error_text[:300], code_style))
                
                story.append(Spacer(1, 0.3*inch))

def convert_notebook_to_pdf(notebook_path, output_path):
    """Convert a Jupyter notebook to PDF"""
    print(f"Converting {notebook_path} to {output_path}...")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Extract practical number and name from filename
    filename = os.path.basename(notebook_path)
    practical_name = filename.replace('.ipynb', '').replace('_', ' ')
    
    # Create PDF
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Add header page
    create_header_page(story, "", practical_name)
    
    # Process notebook cells
    cells = notebook.get('cells', [])
    for cell in cells:
        try:
            process_notebook_cell(cell, story, styles)
        except Exception as e:
            print(f"Error processing cell: {e}")
            continue
    
    # Build PDF
    try:
        doc.build(story)
        print(f"Successfully created {output_path}")
        return True
    except Exception as e:
        print(f"Error building PDF: {e}")
        return False

def main():
    """Main function to convert all notebooks"""
    # Create output directory
    output_dir = "CV_DL_Practicals"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all notebook files
    notebook_files = glob.glob("CVDL_Practical*.ipynb")
    
    if not notebook_files:
        print("No notebook files found!")
        return
    
    print(f"Found {len(notebook_files)} notebooks to convert")
    
    # Convert each notebook
    success_count = 0
    for notebook_file in sorted(notebook_files):
        # Create output filename
        base_name = os.path.splitext(notebook_file)[0]
        output_file = os.path.join(output_dir, f"{base_name}.pdf")
        
        # Convert
        if convert_notebook_to_pdf(notebook_file, output_file):
            success_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted {success_count}/{len(notebook_files)} notebooks")
    print(f"PDFs saved in: {output_dir}")

if __name__ == "__main__":
    main()
