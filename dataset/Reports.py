# classes to convert JSONLs to Python classes
import json
# import pandas as pd

class Block:
    def __init__(self, page_count: int, block_count: int, size: float, font: str, color: int,
                  bbox: list[float], text: list[str], text_str: str) -> None:
        self.page_count = page_count
        self.block_count = block_count
        self.size = size
        self.font = font
        self.color = color
        self.bbox = bbox
        self.text = text
        self.text_str = text_str

    def from_json(json_dict: dict):
        return Block(
            page_count=json_dict['page_count'],
            block_count=json_dict['block_count'],
            size=json_dict['size'],
            font=json_dict['font'],
            color=json_dict['color'],
            bbox=json_dict['bbox'],
            text=json_dict['text'],
            text_str=json_dict['text_str']
        )
    
class Page:
    def __init__(self, page_count: int, block_list: list[Block]) -> None:
        self.page_count = page_count
        self.block_list = block_list # the combined block list make up a page

    def from_json(json_list: list[dict]):
        block_list = []
        page_count = 0
        for block_dict in json_list:
            block = Block.from_json(block_dict)
            block_list.append(block)
        
        return Page(page_count, block_list)

class Report:
    def __init__(self, pages_list: list[Page], company_name: str, industry: str, sector: str, 
                 company_introduction: str, ticker: str, exchange: str, title: str, 
                 date: str, author: str, keywords: str) -> None:
        self.pages_list = pages_list # combined pages list make up a report
        self.company_name = company_name
        self.industry = industry
        self.sector = sector 
        self.company_introduction = company_introduction # short intro of company
        self.ticker = ticker # company ticker symbol
        self.exchange = exchange # exchange the company is listed in
        self.title = title
        self.date = date # just the year
        self.author = author
        self.keywords = keywords
        
    def from_summary(summary_dict: dict, report_dict: dict):
        # summary_dict is the whole summary of a company
        # report_dict is one report from a company's summary
        pages_list = read_normalized_file(report_dict['to_normalization_text_name'])
        return Report(
            pages_list=pages_list,
            company_name=summary_dict['company_name'],
            industry=summary_dict['industry'],
            sector=summary_dict['sector'],
            company_introduction=summary_dict['company_introduction'],
            ticker=summary_dict['ticker'],
            exchange=summary_dict['exchange'],
            title=report_dict['title'] or report_dict['metadata']['title'],
            date=report_dict['date'],
            author=report_dict['metadata']['author'],
            keywords=report_dict['metadata']['keywords']
        )

# reading a JSONL file and converting it to a page list
def read_normalized_file(normalized_filename: str) -> list[Page]:
    # NOTE: each file represents an ESG report, and each line represents a page. 
    # In a page, each dict represents a block, which could be a text block or an image block.
    # The images are all deleted. Each block tells the size, font, and location, of a text.
    with open(f'./normalization_text/{normalized_filename}', encoding="utf8") as json_file:
        json_list = list(json_file)

    page_list: list[Page] = []
    for json_str in json_list:
        page_dict = json.loads(json_str)
        page = Page.from_json(page_dict)
        page_list.append(page)
    
    return page_list

def get_all_reports() -> list[Report]:
        # creating the list of Reports using the summary JSONL and reading all normalized files
        with open(f'./summary.jsonl', encoding="utf8") as json_file:
            summary_json_list = list(json_file)

        report_list = []
        for summary_json_str in summary_json_list:
            summary_dict = json.loads(summary_json_str)
            for report_dict in summary_dict['reports']:
                report = Report.from_summary(summary_dict, report_dict)
                report_list.append(report)
        
        return report_list

def convert_to_dataframe(report_list):
    pass