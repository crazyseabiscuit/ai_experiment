# jiqizhixin/spiders/jiqizhixin_spider.py

import scrapy


class JiqizhixinSpider(scrapy.Spider):
    name = "jiqizhixin"
    allowed_domains = ["www.jiqizhixin.com"]
    start_urls = ["https://www.jiqizhixin.com/"]

    custom_settings = {
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "FEED_FORMAT": "json",
        "FEED_URI": "jiqizhixin_articles.json",
    }

    collected_items = []

    def parse(self, response):
        main_content = response.css("div.u-block__body.home__newest")

        # 遍历每篇文章
        for article in main_content.css("article.article-item__container"):
            title = article.css("a.article-item__title::text").get()
            url = article.css("a.article-item__title::attr(href)").get()
            summary = article.css("p.article-item__summary::text").get()
            author = article.css("a.article-item__name::text").get()
            date = article.css("time.js-time-ago::attr(datetime)").get()
            category = article.css("a.category__link::text").get()

            # 打印提取到的数据，检查是否正确
            print(f"Title: {title}")
            print(f"URL: {url}")
            print(f"Summary: {summary}")
            print(f"Author: {author}")
            print(f"Date: {date}")
            print(f"Category: {category}")
            print("---")

            if url:
                yield {
                    "title": title,
                    "url": response.urljoin(url),
                    "summary": summary,
                    "author": author,
                    "date": date,
                    "category": category,
                }
            else:
                self.logger.warning(f"Skipping article with no URL. Title: {title}")

        next_page = response.css("a.pagination__next::attr(href)").get()
        if next_page:
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse)

    # def closed(self, response):
    #     next_page = response.css("a.next::attr(href)").get()
    #     if next_page:
    #         yield response.follow(next_page, self.parse)

    # def parse_article(self, response):
    #     item = response.meta["item"]
    #     content = "".join(response.css(".single-post-content p::text").getall())
    #     date = response.css(".post-date::text").get()
    #     print(f"Content: {content}, Date: {date}")

    #     item["content"] = content
    #     item["date"] = date
    #     self.collected_items.append(item)
    #     yield item
