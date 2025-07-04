---
layout: post
title: Structure, Layout and Markdown for maintaining this self-notes website
category: [random]
date: 2025-07-04
---

# Markdown

---

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Markdown Elements](#basic-markdown-elements)
3. [Extended Markdown Features](#extended-markdown-features)
4. [Advanced Formatting](#advanced-formatting)
5. [Quick Reference](#quick-reference)

---

## Introduction

Markdown is a lightweight markup language that transforms plain text into beautifully formatted documents. This guide covers everything from basic syntax to advanced features.

> **Note**: This guide follows the [Markdown Crash Course](https://blog.webdevsimplified.com/2023-06/markdown-crash-course/) methodology with enhanced formatting and organization.

---

## Basic Markdown Elements

### Headings: Creating Document Structure

Markdown provides six levels of headings, each serving a specific purpose in document hierarchy:

```markdown
# Primary Title (H1)
## Section Headers (H2)
### Subsection Headers (H3)
#### Minor Headers (H4)
##### Small Headers (H5)
###### Smallest Headers (H6)
```

**Output:**

# Primary Title (H1)
## Section Headers (H2)
### Subsection Headers (H3)
#### Minor Headers (H4)
##### Small Headers (H5)
###### Smallest Headers (H6)

### Paragraphs and Line Breaks

Understanding paragraph formatting is crucial for readable content:

```markdown
This is a standard paragraph. Text flows naturally within paragraph boundaries.

A blank line separates paragraphs, creating distinct content blocks.

For line breaks within paragraphs,  
add two spaces at the end of a line  
to create soft breaks without paragraph separation.
```

**Output:**

This is a standard paragraph. Text flows naturally within paragraph boundaries.

A blank line separates paragraphs, creating distinct content blocks.

For line breaks within paragraphs,  
add two spaces at the end of a line  
to create soft breaks without paragraph separation.

---

## Extended Markdown Features

### Text Styling and Emphasis

Create visual hierarchy and emphasis with various text formatting options:

```markdown
*Italic text* or _italic text_
**Bold text** or __bold text__
***Bold and italic*** or ___bold and italic___
~~Strikethrough text~~
<mark>Highlighted text</mark>
Regular text with <sup>superscript</sup> and <sub>subscript</sub>
```

**Output:**

*Italic text* or _italic text_  
**Bold text** or __bold text__  
***Bold and italic*** or ___bold and italic___  
~~Strikethrough text~~  
<mark>Highlighted text</mark>  
Regular text with <sup>superscript</sup> and <sub>subscript</sub>

### Code Display

#### Inline Code

Use backticks for `inline code` within sentences:

```markdown
Use the `console.log()` function for debugging JavaScript applications.
```

**Output:** Use the `console.log()` function for debugging JavaScript applications.

#### Code Blocks
To display a larger block of code you can wrap your code in three `` ` `` characters. 
- You can also specify the language of your code block by adding the language name after the three `` ` `` characters. 

```javascript
// JavaScript example with syntax highlighting
function greetUser(name) {
    return `Hello, ${name}! Welcome to Markdown.`;
}

const message = greetUser("Developer");
console.log(message);
```

```python
# Python example
def calculate_area(radius):
    """Calculate the area of a circle."""
    import math
    return math.pi * radius ** 2

area = calculate_area(5)
print(f"Circle area: {area:.2f}")
```

---

## Advanced Formatting

### Links and Navigation

Create various types of links for enhanced navigation:

```markdown
[External link](https://blog.webdevsimplified.com)
[Relative link](/2023-06/markdown-crash-course)
[Reference link][1]
<https://direct-url-display.com>

[1]: https://example.com "Reference link tooltip"
```

**Output:**

[External link](https://blog.webdevsimplified.com)  
[Relative link](/2023-06/markdown-crash-course)  
[Reference link][1]  
<https://direct-url-display.com>

[1]: https://example.com "Reference link tooltip"

### Images and Media

```markdown
![Descriptive alt text](/assets/images/google.png "The Google Logo")
```
![The Google Logo](/assets/images/google.png)

### Blockquotes and Citations

Create elegant quotations and nested content:

```markdown
> "The best way to predict the future is to create it."
> — Peter Drucker

> Primary quotation with important information
>> Nested quotation for additional context
>>> Deep nesting for complex hierarchies
```

**Output:**

> "The best way to predict the future is to create it."  
> — Peter Drucker

> Primary quotation with important information
>> Nested quotation for additional context
>>> Deep nesting for complex hierarchies

### Lists and Organization

#### Unordered Lists

```markdown
- **Primary item** with emphasis
- Secondary item
  - Nested sub-item
  - Another sub-item
    - Deep nesting example
- Final primary item
```

**Output:**

- **Primary item** with emphasis
- Secondary item
  - Nested sub-item
  - Another sub-item
    - Deep nesting example
- Final primary item

#### Ordered Lists

```markdown
1. **First step** (numbers auto-increment)
2. Second step with detailed explanation
   1. Sub-step A
   2. Sub-step B
3. Final step
```

**Output:**

1. **First step** (numbers auto-increment)
2. Second step with detailed explanation
   1. Sub-step A
   2. Sub-step B
3. Final step

#### Task Lists

```markdown
- [x] ✅ Completed task
- [x] ✅ Another finished item
- [ ] ⏳ Pending task
- [ ] ⏳ Future task
```

**Output:**

- [x] ✅ Completed task
- [x] ✅ Another finished item
- [ ] ⏳ Pending task
- [ ] ⏳ Future task

### Tables and Data Presentation


- Below the first row you need to add a row where each column consists of at least three `-`s and optionally a `:` character on either side of the `-`s. 
   - The `:` character is used to align the text in the column. 
   - If you add a `:` character on the left side of the `-`s then the text will be left aligned. 
   - If you add a `:` character on the right side of the `-`s then the text will be right aligned. 
   - If you add a `:` character on both sides of the `-`s then the text will be center aligned
- Finally, you can continue to add rows to your table with the same format as your first row.

```markdown
| Feature | Description | Status |
|:--------|:------------|-------:|
| **Basic Syntax** | Core Markdown elements | ✅ Complete |
| **Extended Features** | GitHub Flavored Markdown | ✅ Complete |
| **Advanced Topics** | Complex formatting | 🔄 In Progress |
| **Best Practices** | Professional guidelines | ⏳ Planned |
```

**Output:**

| Feature | Description | Status |
|:--------|:------------|-------:|
| **Basic Syntax** | Core Markdown elements | ✅ Complete |
| **Extended Features** | GitHub Flavored Markdown | ✅ Complete |
| **Advanced Topics** | Complex formatting | 🔄 In Progress |
| **Best Practices** | Professional guidelines | ⏳ Planned |

### Horizontal Rules and Separators

Create visual breaks in your content:

```markdown
Content above separator

---

Content between separators

***

Content below separator
```

**Output:**

Content above separator

---

Content between separators

***

Content below separator

---

## Quick Reference

### Essential Syntax

```markdown
# Headers              → # H1, ## H2, ### H3
*Emphasis*             → *italic*, **bold**, ***both***
`Code`                 → `inline` or ```block```
[Links](url)           → [text](url)
![Images](url)         → ![alt](url)
> Blockquotes          → > text
- Lists                → - item or 1. item
| Tables |             → | col1 | col2 |
---                    → Horizontal rule
```

### GitHub Flavored Markdown

```markdown
~~Strikethrough~~      → ~~text~~
- [ ] Tasks            → - [ ] todo, - [x] done
```

---

## Conclusion

Mastering Markdown enables you to create professional, readable documentation with minimal effort. This guide provides the foundation for beautiful content creation across platforms like GitHub, documentation sites, and blogs.

**Happy writing!** 📝✨

---

*Last updated: July 4, 2025*  
*Version: 1.0*