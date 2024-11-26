from django import template

register = template.Library()


@register.filter
def dict_key(dictionary, key):
    if isinstance(dictionary, dict):
        return dictionary.get(key, '')
    return ''


@register.filter
def get_max(dictionary):
    maximum = -1
    for key in dictionary:
        if dictionary[key] > maximum:
            maximum = dictionary[key]

    return maximum

@register.filter
def nearest_greater_multiple_of_10(number):
    if number % 10 == 0:
        return number
    return (number // 10 + 1) * 10